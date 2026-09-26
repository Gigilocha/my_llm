import copy

import mlflow
import torch
from transformers import PreTrainedTokenizerFast

from src.common.mlflow import setup_mlflow, log_step, log_eval, log_memory, log_speed, log_gradients, log_model_params
from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.dataset import cache_dpo_data, build_dpo_mix_from_disk
from src.data.format import dpo_examples_to_tokenized, tokenize_dpo_example
from src.data.dataloader import create_cycling_tokenized_stream, collect_examples, collate_padded_batch
from src.model.build import build_model
from src.training.loop import run_training_loop
from src.training.dpo_train_step import dpo_train_step, dpo_eval_step
from src.training.optimizer import build_optimizer
from src.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


# Длина ПАРЫ в токенах — chosen и rejected ветки считаются моделью за один
# шаг, суммарная длина — честная мера "стоимости" примера для калибровки
def _dpo_length_fn(tokenizer: PreTrainedTokenizerFast, special_tokens: dict, max_len: int):
    def fn(example: dict) -> int:
        result = tokenize_dpo_example(tokenizer, example, special_tokens, max_len)
        if result is None:
            return 0
        (chosen_ids, _), (rejected_ids, _) = result
        return len(chosen_ids) + len(rejected_ids)
    return fn


# Забрать batch_size пар (chosen, rejected) из потока и собрать ДВА
# независимых padded-батча — по одному на ветку. Специфично для DPO (пары),
# поэтому живёт здесь, а не в dataloader.py — который знает только про
# единичные (ids, labels) примеры, общие для SFT и DPO по отдельности
def _collect_pair_batch(dataloader, batch_size: int, pad_token_id: int):
    pairs = collect_examples(dataloader, batch_size)
    chosen = collate_padded_batch([c for c, _ in pairs], pad_token_id)
    rejected = collate_padded_batch([r for _, r in pairs], pad_token_id)
    return chosen, rejected


def main():
    config = get_config()
    device = resolve_device(config.env.device)
    logger = setup_logger(__name__)
    logger.info(f"Устройство: {get_device_info(device)}")

    monitoring = config.monitoring
    data_dir = PROJECT_ROOT / config.env.data_dir
    rlft_config = config.training.rlft
    dpo_data_cfg = config.data.rlft_data

    setup_mlflow(
        run_name="rlft_dpo",
        tags={"phase": "rlft", "method": "dpo", "device": device},
        params=rlft_config.model_dump(),
    )

    logger.info("Кеширование DPO-данных...")
    cache_dpo_data(dpo_data_cfg, data_dir)

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    special_tokens = config.tokenizer.special_tokens

    # Политика стартует с SFT-чекпоинта — DPO это выравнивание уже обученной
    # отвечать модели, не обучение с нуля
    sft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_sft"
    sft_step = find_latest_checkpoint(sft_checkpoints_dir)
    if sft_step is None:
        raise FileNotFoundError(f"Нет SFT-чекпоинтов в {sft_checkpoints_dir} — сначала запусти sft_train.py")

    policy_model = build_model(config).to(device)
    load_checkpoint(checkpoint_dir=sft_checkpoints_dir, step=sft_step, model=policy_model, device=device)
    logger.info(f"Политика инициализирована из SFT-чекпоинта (шаг {sft_step})")

    # Reference-модель — замороженная КОПИЯ той же стартовой точки. Копируем
    # ДО любых шагов оптимизатора, иначе reference уедет вместе с политикой
    # и регуляризация перестанет работать
    ref_model = copy.deepcopy(policy_model).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    log_model_params(policy_model, logger)
    log_memory(logger, tag="start")

    optimizer = build_optimizer(
        policy_model,
        adamw_lr=rlft_config.learning_rate,
        adamw_weight_decay=rlft_config.weight_decay,
        muon_lr=rlft_config.muon_learning_rate,
        muon_weight_decay=rlft_config.muon_weight_decay,
        muon_momentum=rlft_config.muon_momentum,
    )

    rlft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_rlft"
    latest_rlft_step = find_latest_checkpoint(rlft_checkpoints_dir)
    if latest_rlft_step is not None:
        load_checkpoint(checkpoint_dir=rlft_checkpoints_dir, step=latest_rlft_step, model=policy_model, optimizer=optimizer, device=device)
        start_step = latest_rlft_step + 1
        logger.info(f"Восстановлено DPO-обучение с шага {start_step}")
    else:
        start_step = 0

    compiled_policy = torch.compile(policy_model) if device == "cuda" else policy_model
    compiled_ref = torch.compile(ref_model) if device == "cuda" else ref_model

    length_fn = _dpo_length_fn(tokenizer, special_tokens, rlft_config.max_len)
    tokenize_fn = lambda examples: dpo_examples_to_tokenized(examples, tokenizer, special_tokens, rlft_config.max_len)

    train_mix_factory = lambda: build_dpo_mix_from_disk(dpo_data_cfg, data_dir, split="train", length_fn=length_fn)
    train_dataloader = create_cycling_tokenized_stream(train_mix_factory, tokenize_fn)

    val_mix_factory = lambda: build_dpo_mix_from_disk(dpo_data_cfg, data_dir, split="val", length_fn=length_fn)
    val_dataloader = create_cycling_tokenized_stream(val_mix_factory, tokenize_fn)

    def micro_step() -> tuple[float, dict]:
        (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = _collect_pair_batch(
            train_dataloader, rlft_config.batch_size, tokenizer.pad_token_id
        )
        c_ids, c_labels, c_mask = c_ids.to(device), c_labels.to(device), c_mask.to(device)
        r_ids, r_labels, r_mask = r_ids.to(device), r_labels.to(device), r_mask.to(device)

        loss, metrics = dpo_train_step(
            compiled_policy, compiled_ref,
            c_ids, c_labels, c_mask, r_ids, r_labels, r_mask,
            beta=rlft_config.beta, grad_accum_steps=rlft_config.gradient_accumulation_steps,
        )
        return loss, metrics

    def on_step(step: int, avg_loss: float, lr: float, grad_norm: float, elapsed: float, metrics: dict) -> None:
        log_step(logger, step, avg_loss, lr, grad_norm)
        logger.info(f"step {step}, accuracy={metrics['accuracy']:.3f}, margin={metrics['reward_margin']:.4f}")
        mlflow.log_metric("dpo_accuracy", metrics["accuracy"], step=step)
        mlflow.log_metric("reward_margin", metrics["reward_margin"], step=step)
        if step % monitoring.speed_monitoring.interval_steps == 0:
            log_speed(logger, step, rlft_config.batch_size, rlft_config.max_len, elapsed, rlft_config.gradient_accumulation_steps)
        if step % monitoring.memory_monitoring.interval_steps == 0:
            log_memory(logger, step, tag="training")
        if step % monitoring.gradient_monitoring.interval_steps == 0:
            log_gradients(policy_model, step, logger)

    def checkpoint_fn(step: int, avg_loss: float) -> None:
        save_checkpoint(policy_model, optimizer, step, avg_loss, rlft_checkpoints_dir)
        logger.info(f"DPO-чекпоинт сохранён на шаге {step}")

    def eval_fn(step: int) -> None:
        (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = _collect_pair_batch(
            val_dataloader, rlft_config.batch_size, tokenizer.pad_token_id
        )
        c_ids, c_labels, c_mask = c_ids.to(device), c_labels.to(device), c_mask.to(device)
        r_ids, r_labels, r_mask = r_ids.to(device), r_labels.to(device), r_mask.to(device)
        val_loss, val_acc = dpo_eval_step(
            compiled_policy, compiled_ref, c_ids, c_labels, c_mask, r_ids, r_labels, r_mask, beta=rlft_config.beta,
        )
        log_eval(logger, step, val_loss)
        logger.info(f"step {step}, val_accuracy={val_acc:.3f}")
        mlflow.log_metric("val_dpo_accuracy", val_acc, step=step)

    run_training_loop(
        start_step=start_step,
        max_steps=rlft_config.max_steps,
        warmup_steps=rlft_config.warmup_steps,
        learning_rate=rlft_config.learning_rate,
        min_learning_rate=rlft_config.min_learning_rate,
        grad_clip_norm=rlft_config.grad_clip_norm,
        checkpoint_interval=rlft_config.checkpoint_interval,
        eval_interval=rlft_config.eval_interval,
        grad_accum_steps=rlft_config.gradient_accumulation_steps,
        optimizer=optimizer,
        model_for_clip=policy_model,
        micro_step=micro_step,
        on_step=on_step,
        checkpoint_fn=checkpoint_fn,
        eval_fn=eval_fn,
    )

    mlflow.end_run()


if __name__ == "__main__":
    main()