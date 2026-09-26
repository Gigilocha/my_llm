import argparse
import math

import mlflow
from transformers import PreTrainedTokenizerFast

from src.common.mlflow import setup_mlflow
from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.dataset import build_language_mix_from_disk, extract_texts
from src.data.dataloader import create_cycling_pretrain_dataloader, collate_pretrain_batch
from src.model.build import build_model
from src.training.checkpoint import find_latest_checkpoint, load_checkpoint
from src.training.eval_step import eval_step
from src.engine.select import make_generate_fn


# Промпты для качественной проверки — по одному-два на язык, разной сложности
GENERATION_PROMPTS = {
    "rus": [
        "Литва — это",
        "В новостях сегодня сообщили, что",
    ],
    "en": [
        "The history of artificial intelligence begins",
        "In today's news, scientists announced",
    ],
    "code": [
        "def sum(a, b):",
        "class LinkedList:",
    ],
}


# Честная оценка по одному языку: усредняем loss по num_batches независимым
# батчам (одного батча, как раньше в base_train.py, недостаточно — слишком шумно,
# см. "Финальный val_loss" на одном батче, который оказался неожиданно низким).
# create_cycling_pretrain_dataloader пересоздаёт стрим на пересечении конца данных —
# val маленький, на num_batches его может не хватить одним проходом
def eval_language(
    model, tokenizer, language: str, data_dir, config, device: str, num_batches: int, batch_size: int, max_len: int,
) -> tuple[float, float]:
    sources = getattr(config.data.pre_training_data, f"{language}_sources")

    text_factory = lambda: extract_texts(
        build_language_mix_from_disk(sources, "pretrain", language, data_dir, config.data.pre_training_data.seed, split="val")
    )
    dataloader = create_cycling_pretrain_dataloader(text_factory, tokenizer, max_len)

    losses = []
    for _ in range(num_batches):
        batch = collate_pretrain_batch(dataloader, batch_size).to(device)
        losses.append(eval_step(model, batch))

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Оценка чекпоинта: loss по языкам + примеры генерации")
    parser.add_argument("--step", type=int, default=None, help="Номер шага чекпоинта (по умолчанию — последний доступный)")
    parser.add_argument("--num-eval-batches", type=int, default=50, help="Сколько val-батчей усреднять на язык")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="По умолчанию — из engine_config.yaml")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    args = parser.parse_args()

    config = get_config()
    device = resolve_device(config.env.device)
    logger = setup_logger(__name__)
    logger.info(f"Устройство: {get_device_info(device)}")

    # CLI-флаги — необязательные оверрайды поверх engine_config.yaml, не
    # единственный источник значений (если не передан — берём из конфига)
    engine_cfg = config.engine.model_copy(update={
        k: v for k, v in {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }.items() if v is not None
    })

    data_dir = PROJECT_ROOT / config.env.data_dir
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"

    step = args.step if args.step is not None else find_latest_checkpoint(checkpoints_dir)
    if step is None:
        raise FileNotFoundError(f"Нет чекпоинтов в {checkpoints_dir}")
    logger.info(f"Оцениваю чекпоинт на шаге {step}")

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    model = build_model(config)
    model = model.to(device)
    load_checkpoint(checkpoint_dir=checkpoints_dir, step=step, model=model, device=device)  # optimizer не нужен для оценки
    model.eval()

    pretrain_config = config.training.pre_training

    # --- Количественная оценка: loss/perplexity по каждому языку отдельно ---
    logger.info(f"Считаю val_loss по {args.num_eval_batches} батчам на язык...")
    results = {}
    for language in ("rus", "en", "code"):
        avg_loss, perplexity = eval_language(
            model, tokenizer, language, data_dir, config, device,
            num_batches=args.num_eval_batches,
            batch_size=pretrain_config.batch_size,
            max_len=pretrain_config.max_len,
        )
        results[language] = (avg_loss, perplexity)
        logger.info(f"  {language}: val_loss={avg_loss:.4f}, perplexity={perplexity:.2f}")

    overall_loss = sum(loss for loss, _ in results.values()) / len(results)
    overall_perplexity = math.exp(overall_loss)
    logger.info(f"Среднее по языкам: val_loss={overall_loss:.4f}, perplexity={overall_perplexity:.2f}")

    # Логируем в MLflow отдельным run, привязанным к шагу чекпоинта —
    # чтобы можно было сравнивать разные чекпоинты между собой со временем.
    # Тот же tracking URI, что и в base_train.py (setup_mlflow), без явного
    # tracking_uri MLflow создаёт mlruns/ там, откуда запущен скрипт
    # (обычно корень репозитория), а не в outputs/mlflow/
    setup_mlflow(run_name=f"eval_step_{step}")
    mlflow.log_param("checkpoint_step", step)
    for language, (loss, perplexity) in results.items():
        mlflow.log_metric(f"val_loss_{language}", loss)
        mlflow.log_metric(f"perplexity_{language}", perplexity)
    mlflow.log_metric("val_loss_overall", overall_loss)
    mlflow.log_metric("perplexity_overall", overall_perplexity)

    # --- Качественная оценка: реальные генерации, чтобы прочитать глазами ---
    logger.info(f"Генерирую примеры (движок: {'GenerationEngine/KV-кеш' if engine_cfg.use_kv_cache else 'naive'})...")
    generate_fn = make_generate_fn(model, tokenizer, device, config.model.model.max_position_embeddings, engine_cfg)
    generation_log = []
    for language, prompts in GENERATION_PROMPTS.items():
        for prompt in prompts:
            text = generate_fn(prompt)
            logger.info(f"[{language}] промпт: {prompt!r}")
            logger.info(f"[{language}] генерация: {text!r}")
            generation_log.append(f"=== {language} ===\nПромпт: {prompt}\nГенерация: {text}\n")

    mlflow.log_text("\n".join(generation_log), "generations.txt")
    mlflow.end_run()


if __name__ == "__main__":
    main()