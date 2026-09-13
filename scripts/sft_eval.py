import argparse
import math
import os

import mlflow
from transformers import PreTrainedTokenizerFast

from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.sft_dataset import build_sft_mix_from_disk
from src.data.sft_format import sft_examples_to_tokenized, format_prompt_for_generation
from src.data.sft_dataloader import collate_sft_batch
from src.model.transformer import Transformer
from src.training.checkpoint import find_latest_checkpoint, load_checkpoint
from src.training.sft_train_step import sft_eval_step
from src.engine.select import make_generate_fn


# Тестовые инструкции для качественной проверки — по языку/домену. В отличие
# от base_eval.py (продолжение сырого текста), тут промпт форматируется ТАК ЖЕ,
# как при обучении (format_sft_texts) — модель должна отвечать на вопрос,
# а не просто продолжать предложение
TEST_INSTRUCTIONS = {
    "rus": [
        {"messages": [{"role": "user", "content": "Объясни, что такое рекурсия, простыми словами."}]},
        {"messages": [{"role": "user", "content": "Напиши короткое стихотворение про осень."}]},
    ],
    "en": [
        {"messages": [{"role": "user", "content": "Explain the difference between a list and a tuple in Python."}]},
        {"messages": [{"role": "user", "content": "What are the main causes of climate change?"}]},
    ],
    "code": [
        {"messages": [{"role": "user", "content": "Write a Python function that checks if a number is prime."}]},
        {"messages": [{"role": "user", "content": "Напиши функцию сортировки пузырьком на Python."}]},
    ],
}


def build_model(config) -> Transformer:
    return Transformer(
        vocab_size=config.model.model.vocab_size,
        num_layer=config.model.model.num_layers,
        hidden_size=config.model.model.hidden_size,
        head_dim=config.model.attention.head_dim,
        num_heads=config.model.attention.num_heads,
        num_kv_heads=config.model.attention.num_kv_heads,
        use_qk_norm=config.model.attention.use_qk_norm,
        qk_norm_eps=config.model.attention.qk_norm_eps,
        rope_theta=config.model.attention.rope_theta,
        max_position_embeddings=config.model.model.max_position_embeddings,
        intermediate_size=config.model.mlp.intermediate_size,
        norm_eps=config.model.model.norm_eps,
    )


def main():
    parser = argparse.ArgumentParser(description="Оценка SFT-чекпоинта: loss на val + генерация по реальным инструкциям")
    parser.add_argument("--step", type=int, default=None, help="Номер шага SFT-чекпоинта (по умолчанию — последний)")
    parser.add_argument("--num-eval-batches", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    args = parser.parse_args()

    config = get_config()
    device = resolve_device(config.env.device)
    logger = setup_logger(__name__)
    logger.info(f"Устройство: {get_device_info(device)}")

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
    sft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_sft"

    step = args.step if args.step is not None else find_latest_checkpoint(sft_checkpoints_dir)
    if step is None:
        raise FileNotFoundError(f"Нет SFT-чекпоинтов в {sft_checkpoints_dir} — сначала запусти base_sft.py")
    logger.info(f"Оцениваю SFT-чекпоинт на шаге {step}")

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    special_tokens = config.tokenizer.special_tokens

    model = build_model(config).to(device)
    load_checkpoint(checkpoint_dir=sft_checkpoints_dir, step=step, model=model)  # optimizer не нужен для оценки
    model.eval()

    sft_config = config.training.sft
    sft_data_cfg = config.data.sft_data

    # --- Количественная оценка: усреднённый val_loss на замаскированном лоссе,
    # как при обучении (только ответ, не промпт) ---
    logger.info(f"Считаю SFT val_loss по {args.num_eval_batches} батчам...")

    def val_examples():
        return build_sft_mix_from_disk(sft_data_cfg, data_dir, split="val")

    tokenized_stream = sft_examples_to_tokenized(val_examples(), tokenizer, special_tokens, sft_config.max_len)

    losses = []
    batch_buffer = []
    for tokenized in tokenized_stream:
        batch_buffer.append(tokenized)
        if len(batch_buffer) == sft_config.batch_size:
            input_ids, labels, attention_mask = collate_sft_batch(batch_buffer, tokenizer.pad_token_id)
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
            losses.append(sft_eval_step(model, input_ids, labels, attention_mask))
            batch_buffer = []
        if len(losses) >= args.num_eval_batches:
            break

    if not losses:
        raise RuntimeError("Не набралось ни одного полного val-батча — уменьши --num-eval-batches или batch_size")

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    logger.info(f"SFT val_loss={avg_loss:.4f}, perplexity={perplexity:.2f} (по {len(losses)} батчам)")

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_dir = PROJECT_ROOT / "outputs" / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlflow_dir}/mlruns")

    mlflow.start_run(run_name=f"sft_eval_step_{step}")
    mlflow.log_param("checkpoint_step", step)
    mlflow.log_metric("sft_val_loss", avg_loss)
    mlflow.log_metric("sft_perplexity", perplexity)

    # --- Качественная оценка: РЕАЛЬНЫЕ инструкции, формат как при обучении ---
    logger.info(f"Генерирую ответы на тестовые инструкции (движок: {'GenerationEngine/KV-кеш' if engine_cfg.use_kv_cache else 'naive'})...")
    generate_fn = make_generate_fn(model, tokenizer, device, config.model.model.max_position_embeddings, engine_cfg)

    generation_log = []
    for domain, examples in TEST_INSTRUCTIONS.items():
        for example in examples:
            instruction_text = example["messages"][0]["content"]
            prompt_text = format_prompt_for_generation(example["messages"], special_tokens)
            response = generate_fn(prompt_text)
            logger.info(f"[{domain}] инструкция: {instruction_text!r}")
            logger.info(f"[{domain}] ответ: {response!r}")
            generation_log.append(f"=== {domain} ===\nИнструкция: {instruction_text}\nОтвет: {response}\n")

    mlflow.log_text("\n".join(generation_log), "sft_generations.txt")
    mlflow.end_run()


if __name__ == "__main__":
    main()
