import argparse

import mlflow
from transformers import PreTrainedTokenizerFast

from src.common.mlflow import setup_mlflow
from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.dataset import build_dpo_mix_from_disk
from src.data.format import dpo_examples_to_tokenized, format_prompt_for_generation
from src.data.dataloader import collate_padded_batch
from src.model.build import build_model
from src.training.checkpoint import find_latest_checkpoint, load_checkpoint
from src.training.dpo_train_step import dpo_eval_step
from src.engine.select import make_generate_fn


"""
Оценка DPO-чекпоинта: val accuracy/loss (политика vs reference), плюс
качественная генерация политики на тестовых инструкциях.

Reference-модель для оценки — тот же SFT-чекпоинт, что rlft_train.py
использовал как точку старта DPO (см. copy.deepcopy(policy_model) в начале
обучения там). Если --ref-step явно не передан, берётся ПОСЛЕДНИЙ
SFT-чекпоинт на диске — то же допущение, что делает rlft_train.py при
инициализации. Если SFT переобучался ПОСЛЕ старта DPO, "последний
SFT-чекпоинт" может не совпадать с тем, что реально было заморожено как
reference — в этом случае передай --ref-step явно.
"""

TEST_INSTRUCTIONS = {
    "rus": [
        {"messages": [{"role": "user", "content": "Объясни, что такое рекурсия, простыми словами."}]},
    ],
    "en": [
        {"messages": [{"role": "user", "content": "What are the main causes of climate change?"}]},
    ],
    "code": [
        {"messages": [{"role": "user", "content": "Write a Python function that checks if a number is prime."}]},
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Оценка DPO-чекпоинта: val accuracy/loss + генерация")
    parser.add_argument("--step", type=int, default=None, help="Номер шага DPO-чекпоинта (по умолчанию — последний)")
    parser.add_argument("--ref-step", type=int, default=None, help="Номер шага SFT-чекпоинта для reference (по умолчанию — последний)")
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
    rlft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_rlft"
    sft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_sft"

    step = args.step if args.step is not None else find_latest_checkpoint(rlft_checkpoints_dir)
    if step is None:
        raise FileNotFoundError(f"Нет DPO-чекпоинтов в {rlft_checkpoints_dir} — сначала запусти rlft_train.py")

    ref_step = args.ref_step if args.ref_step is not None else find_latest_checkpoint(sft_checkpoints_dir)
    if ref_step is None:
        raise FileNotFoundError(f"Нет SFT-чекпоинтов в {sft_checkpoints_dir} — нечего использовать как reference")

    logger.info(f"Оцениваю DPO-чекпоинт на шаге {step}, reference — SFT-чекпоинт на шаге {ref_step}")

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    special_tokens = config.tokenizer.special_tokens

    policy_model = build_model(config).to(device)
    load_checkpoint(checkpoint_dir=rlft_checkpoints_dir, step=step, model=policy_model, device=device)
    policy_model.eval()

    ref_model = build_model(config).to(device)
    load_checkpoint(checkpoint_dir=sft_checkpoints_dir, step=ref_step, model=ref_model, device=device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    rlft_config = config.training.rlft
    dpo_data_cfg = config.data.rlft_data

    # --- Количественная оценка: усреднённая accuracy/loss по val-парам ---
    logger.info(f"Считаю DPO val_loss/accuracy по {args.num_eval_batches} батчам...")

    def val_examples():
        return build_dpo_mix_from_disk(dpo_data_cfg, data_dir, split="val")

    tokenized_stream = dpo_examples_to_tokenized(val_examples(), tokenizer, special_tokens, rlft_config.max_len)

    losses, accuracies = [], []
    batch_buffer = []
    for tokenized in tokenized_stream:
        batch_buffer.append(tokenized)
        if len(batch_buffer) == rlft_config.batch_size:
            chosen = collate_padded_batch([c for c, _ in batch_buffer], tokenizer.pad_token_id)
            rejected = collate_padded_batch([r for _, r in batch_buffer], tokenizer.pad_token_id)
            c_ids, c_labels, c_mask = (t.to(device) for t in chosen)
            r_ids, r_labels, r_mask = (t.to(device) for t in rejected)

            val_loss, val_acc = dpo_eval_step(
                policy_model, ref_model, c_ids, c_labels, c_mask, r_ids, r_labels, r_mask, beta=rlft_config.beta,
            )
            losses.append(val_loss)
            accuracies.append(val_acc)
            batch_buffer = []
        if len(losses) >= args.num_eval_batches:
            break

    if not losses:
        raise RuntimeError("Не набралось ни одного полного val-батча — уменьши --num-eval-batches или batch_size")

    avg_loss = sum(losses) / len(losses)
    avg_accuracy = sum(accuracies) / len(accuracies)
    logger.info(f"DPO val_loss={avg_loss:.4f}, val_accuracy={avg_accuracy:.3f} (по {len(losses)} батчам)")

    setup_mlflow(run_name=f"rlft_eval_step_{step}")
    mlflow.log_param("checkpoint_step", step)
    mlflow.log_param("ref_step", ref_step)
    mlflow.log_metric("dpo_val_loss", avg_loss)
    mlflow.log_metric("dpo_val_accuracy", avg_accuracy)

    # --- Качественная оценка: генерация политики на тестовых инструкциях ---
    logger.info(f"Генерирую ответы политики (движок: {'GenerationEngine/KV-кеш' if engine_cfg.use_kv_cache else 'naive'})...")
    generate_fn = make_generate_fn(policy_model, tokenizer, device, config.model.model.max_position_embeddings, engine_cfg)

    generation_log = []
    for domain, examples in TEST_INSTRUCTIONS.items():
        for example in examples:
            instruction_text = example["messages"][0]["content"]
            prompt_text = format_prompt_for_generation(example["messages"], special_tokens)
            response = generate_fn(prompt_text)
            logger.info(f"[{domain}] инструкция: {instruction_text!r}")
            logger.info(f"[{domain}] ответ: {response!r}")
            generation_log.append(f"=== {domain} ===\nИнструкция: {instruction_text}\nОтвет: {response}\n")

    mlflow.log_text("\n".join(generation_log), "rlft_generations.txt")
    mlflow.end_run()


if __name__ == "__main__":
    main()