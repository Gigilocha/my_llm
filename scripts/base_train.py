import mlflow
from transformers import PreTrainedTokenizerFast
import torch

from src.common.mlflow import setup_mlflow, log_step, log_eval, log_memory, log_speed, log_gradients, log_model_params
from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.dataset import build_pretrain_mix_from_disk, extract_texts
from src.data.dataloader import create_cycling_pretrain_dataloader, collate_pretrain_batch
from src.model.build import build_model
from src.tokenizer.tokenizer import encode
from src.training.loop import run_training_loop
from src.training.train_step import train_step
from src.training.eval_step import eval_step
from src.training.optimizer import build_optimizer
from src.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


def main():
    config = get_config()
    device = resolve_device(config.env.device)
    logger = setup_logger(__name__)
    logger.info(f"Устройство: {get_device_info(device)}")

    monitoring = config.monitoring
    pretrain_config = config.training.pre_training

    setup_mlflow(
        run_name="pretrain",
        tags={"model_type": "transformer", "dataset": "pretrain_mix", "language": "rus/en/code", "device": device},
        params={
            **config.model.model.model_dump(),
            **config.model.attention.model_dump(),
            **config.model.mlp.model_dump(),
            **pretrain_config.model_dump(),
        },
    )

    data_dir = PROJECT_ROOT / config.env.data_dir
    pretrain_data_cfg = config.data.pre_training_data

    logger.info("Инициализация токенизатора")
    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    # Токенная калибровка (rus/en/code_quantity -> реальная доля ТОКЕНОВ) —
    # та же мера длины, что увидит обучение
    token_length_fn = lambda text: len(encode(tokenizer, text, add_special_tokens=False))

    logger.info("Сборка тренировочных данных (train)...")
    train_text_factory = lambda: extract_texts(
        build_pretrain_mix_from_disk(pretrain_data_cfg, data_dir, split="train", length_fn=token_length_fn)
    )
    pretrain_dataloader = create_cycling_pretrain_dataloader(train_text_factory, tokenizer, pretrain_config.max_len)

    logger.info("Сборка валидационных данных (val)...")
    val_text_factory = lambda: extract_texts(
        build_pretrain_mix_from_disk(pretrain_data_cfg, data_dir, split="val", length_fn=token_length_fn)
    )
    val_dataloader = create_cycling_pretrain_dataloader(val_text_factory, tokenizer, pretrain_config.max_len)

    logger.info("Инициализация модели GPT")
    model = build_model(config).to(device)
    log_model_params(model, logger)
    log_memory(logger, tag="start")

    optimizer = build_optimizer(
        model,
        adamw_lr=pretrain_config.learning_rate,
        adamw_weight_decay=pretrain_config.weight_decay,
        muon_lr=pretrain_config.muon_learning_rate,
        muon_weight_decay=pretrain_config.muon_weight_decay,
        muon_momentum=pretrain_config.muon_momentum,
    )

    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    latest_step = find_latest_checkpoint(checkpoints_dir)
    if latest_step is not None:
        load_checkpoint(checkpoint_dir=checkpoints_dir, step=latest_step, model=model, optimizer=optimizer, device=device)
        start_step = latest_step + 1
        logger.info(f"Восстановлено обучение с шага {start_step}")
    else:
        start_step = 0

    # Компилируем ПОСЛЕ загрузки чекпоинта, сохраняем всегда через `model`
    # (некомпилированный) — чекпоинты остаются переносимыми независимо от
    # того, был ли включён compile в конкретном запуске
    compiled_model = torch.compile(model) if device == "cuda" else model

    def micro_step() -> tuple[float, dict]:
        batch = collate_pretrain_batch(pretrain_dataloader, pretrain_config.batch_size).to(device)
        loss = train_step(compiled_model, batch, pretrain_config.gradient_accumulation_steps)
        return loss, {}

    def on_step(step: int, avg_loss: float, lr: float, grad_norm: float, elapsed: float, metrics: dict) -> None:
        log_step(logger, step, avg_loss, lr, grad_norm)
        if step % monitoring.speed_monitoring.interval_steps == 0:
            log_speed(logger, step, pretrain_config.batch_size, pretrain_config.max_len, elapsed, pretrain_config.gradient_accumulation_steps)
        if step % monitoring.memory_monitoring.interval_steps == 0:
            log_memory(logger, step, tag="training")
        if step % monitoring.gradient_monitoring.interval_steps == 0:
            log_gradients(model, step, logger)

    def checkpoint_fn(step: int, avg_loss: float) -> None:
        save_checkpoint(model, optimizer, step, avg_loss, checkpoints_dir)
        logger.info(f"Чекпоинт сохранён на шаге {step}")

    def eval_fn(step: int) -> None:
        val_batch = collate_pretrain_batch(val_dataloader, pretrain_config.batch_size).to(device)
        val_loss = eval_step(compiled_model, val_batch)
        log_eval(logger, step, val_loss)

    run_training_loop(
        start_step=start_step,
        max_steps=pretrain_config.max_steps,
        warmup_steps=pretrain_config.warmup_steps,
        learning_rate=pretrain_config.learning_rate,
        min_learning_rate=pretrain_config.min_learning_rate,
        grad_clip_norm=pretrain_config.grad_clip_norm,
        checkpoint_interval=pretrain_config.checkpoint_interval,
        eval_interval=pretrain_config.eval_interval,
        grad_accum_steps=pretrain_config.gradient_accumulation_steps,
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