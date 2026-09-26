import mlflow
from transformers import PreTrainedTokenizerFast
import torch

from src.common.mlflow import setup_mlflow, log_step, log_eval, log_memory, log_speed, log_gradients, log_model_params
from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.dataset import cache_sft_data, build_sft_mix_from_disk
from src.data.format import sft_examples_to_tokenized, tokenize_sft_example
from src.data.dataloader import create_cycling_tokenized_stream, collect_examples, collate_padded_batch
from src.model.build import build_model
from src.training.loop import run_training_loop
from src.training.sft_train_step import sft_train_step, sft_eval_step
from src.training.optimizer import build_optimizer
from src.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


# Длина ОТФОРМАТИРОВАННОГО примера в токенах — та же величина, что реально
# увидит обучение (со спецтокенами, после обрезки по max_len), не сырая
# длина текста. Пример, не прошедший токенизацию (None), считается длиной 0
def _sft_length_fn(tokenizer: PreTrainedTokenizerFast, special_tokens: dict, max_len: int):
    def fn(example: dict) -> int:
        result = tokenize_sft_example(tokenizer, example, special_tokens, max_len)
        return len(result[0]) if result is not None else 0
    return fn


def main():
    config = get_config()
    device = resolve_device(config.env.device)
    logger = setup_logger(__name__)
    logger.info(f"Устройство: {get_device_info(device)}")

    monitoring = config.monitoring
    data_dir = PROJECT_ROOT / config.env.data_dir
    sft_config = config.training.sft
    sft_data_cfg = config.data.sft_data

    setup_mlflow(run_name="sft", tags={"phase": "sft", "device": device}, params=sft_config.model_dump())

    logger.info("Кеширование SFT-данных...")
    cache_sft_data(sft_data_cfg, data_dir)

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    special_tokens = config.tokenizer.special_tokens

    model = build_model(config).to(device)

    # Стартуем с последнего pretrain-чекпоинта — SFT это дообучение, не
    # обучение с нуля. optimizer=None: веса грузим, а optimizer-состояние
    # pretrain для SFT не нужно — оптимизатор SFT собираем заново
    pretrain_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    pretrain_step = find_latest_checkpoint(pretrain_checkpoints_dir)
    if pretrain_step is None:
        raise FileNotFoundError(f"Нет pretrain-чекпоинтов в {pretrain_checkpoints_dir} — сначала обучи базовую модель")
    load_checkpoint(checkpoint_dir=pretrain_checkpoints_dir, step=pretrain_step, model=model, device=device)
    logger.info(f"Загружены веса pretrain-чекпоинта (шаг {pretrain_step})")

    log_model_params(model, logger)
    log_memory(logger, tag="start")

    optimizer = build_optimizer(
        model,
        adamw_lr=sft_config.learning_rate,
        adamw_weight_decay=sft_config.weight_decay,
        muon_lr=sft_config.muon_learning_rate,
        muon_weight_decay=sft_config.muon_weight_decay,
        muon_momentum=sft_config.muon_momentum,
    )

    sft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_sft"
    latest_sft_step = find_latest_checkpoint(sft_checkpoints_dir)
    if latest_sft_step is not None:
        load_checkpoint(checkpoint_dir=sft_checkpoints_dir, step=latest_sft_step, model=model, optimizer=optimizer, device=device)
        start_step = latest_sft_step + 1
        logger.info(f"Восстановлено SFT-обучение с шага {start_step}")
    else:
        start_step = 0

    compiled_model = torch.compile(model) if device == "cuda" else model

    length_fn = _sft_length_fn(tokenizer, special_tokens, sft_config.max_len)
    tokenize_fn = lambda examples: sft_examples_to_tokenized(examples, tokenizer, special_tokens, sft_config.max_len)

    train_mix_factory = lambda: build_sft_mix_from_disk(sft_data_cfg, data_dir, split="train", length_fn=length_fn)
    train_dataloader = create_cycling_tokenized_stream(train_mix_factory, tokenize_fn)

    val_mix_factory = lambda: build_sft_mix_from_disk(sft_data_cfg, data_dir, split="val", length_fn=length_fn)
    val_dataloader = create_cycling_tokenized_stream(val_mix_factory, tokenize_fn)

    def micro_step() -> tuple[float, dict]:
        examples = collect_examples(train_dataloader, sft_config.batch_size)
        input_ids, labels, attention_mask = collate_padded_batch(examples, tokenizer.pad_token_id)
        input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
        loss = sft_train_step(compiled_model, input_ids, labels, attention_mask, sft_config.gradient_accumulation_steps)
        return loss, {}

    def on_step(step: int, avg_loss: float, lr: float, grad_norm: float, elapsed: float, metrics: dict) -> None:
        log_step(logger, step, avg_loss, lr, grad_norm)
        if step % monitoring.speed_monitoring.interval_steps == 0:
            log_speed(logger, step, sft_config.batch_size, sft_config.max_len, elapsed, sft_config.gradient_accumulation_steps)
        if step % monitoring.memory_monitoring.interval_steps == 0:
            log_memory(logger, step, tag="training")
        if step % monitoring.gradient_monitoring.interval_steps == 0:
            log_gradients(model, step, logger)

    def checkpoint_fn(step: int, avg_loss: float) -> None:
        save_checkpoint(model, optimizer, step, avg_loss, sft_checkpoints_dir)
        logger.info(f"SFT-чекпоинт сохранён на шаге {step}")

    def eval_fn(step: int) -> None:
        val_examples = collect_examples(val_dataloader, sft_config.batch_size)
        v_ids, v_labels, v_mask = collate_padded_batch(val_examples, tokenizer.pad_token_id)
        v_ids, v_labels, v_mask = v_ids.to(device), v_labels.to(device), v_mask.to(device)
        val_loss = sft_eval_step(compiled_model, v_ids, v_labels, v_mask)
        log_eval(logger, step, val_loss)

    run_training_loop(
        start_step=start_step,
        max_steps=sft_config.max_steps,
        warmup_steps=sft_config.warmup_steps,
        learning_rate=sft_config.learning_rate,
        min_learning_rate=sft_config.min_learning_rate,
        grad_clip_norm=sft_config.grad_clip_norm,
        checkpoint_interval=sft_config.checkpoint_interval,
        eval_interval=sft_config.eval_interval,
        grad_accum_steps=sft_config.gradient_accumulation_steps,
        optimizer=optimizer,
        model_for_clip=model,
        micro_step=micro_step,
        on_step=on_step,
        checkpoint_fn=checkpoint_fn,
        eval_fn=eval_fn,
    )

    mlflow.end_run()


if __name__ == "__main__":
    main()