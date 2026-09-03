from pathlib import Path
from transformers import PreTrainedTokenizerFast
import torch
import mlflow
import os
import time

from src.common.mlflow import (
    log_step, log_eval, log_memory, log_speed, 
    log_gradients, log_model_params, log_artifacts, log_model
)
from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import build_pretrain_mix_from_disk, extract_texts
from src.data.dataloader import (
    create_pretrain_dataloader,
    create_cycling_pretrain_dataloader,
    collate_pretrain_batch,
)
from src.model.transformer import Transformer
from src.training.train_step import train_step
from src.training.eval_step import eval_step
from src.training.optimizer import build_optimizer
from src.training.lr_schedule import get_lr
from src.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


def main():
    # Конфиг
    config = get_config()

    # Устройство
    device = config.env.device

    # Логгер
    logger = setup_logger(__name__)

    # Настройка mlflow
    # Разрешаем filestore
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # Устанавливаем путь
    mlflow_dir = PROJECT_ROOT / "outputs" / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    # Устанавливает трекинг URI
    mlflow.set_tracking_uri(f"file:///{mlflow_dir}/mlruns")

    # Тэги
    mlflow.set_tags({
        "model_type": "transformer",
        "dataset": "pretrain_mix",
        "language": "rus/en/code",
        "device": config.env.device,
    })

    # Запуск MLflow
    mlflow.start_run(run_name="pretrain")

    params = {
        **config.model.model.model_dump(),           # ModelConfig
        **config.model.attention.model_dump(),       # AttentionConfig
        **config.model.mlp.model_dump(),             # MLPConfig
        **config.training.pre_training.model_dump(), # PretrainingConfig
    }
    mlflow.log_params(params)
    
    # Директория данных
    data_dir = PROJECT_ROOT / config.env.data_dir
    pretrain_data_cfg = config.data.pre_training_data

    # Инициализация токенизатора
    logger.info("Инициализация токенизатора")
    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    # Сбор тренировочного микса для Pre-training (train) — один проход по всему train-корпусу
    logger.info("Сборка тренировочных данных (train)...")
    train_mix = build_pretrain_mix_from_disk(pretrain_data_cfg, data_dir, split="train")
    train_text = extract_texts(train_mix)
    pretrain_dataloader = create_pretrain_dataloader(
        train_text,
        tokenizer,
        config.training.pre_training.max_len
    )

    # Оценочные данные для Pre-training (val) — данных мало (val_split_ratio),
    # поэтому стрим пересоздаётся заново каждый раз, когда заканчивается
    logger.info("Сборка валидационных данных (val)...")
    val_text_factory = lambda: extract_texts(
        build_pretrain_mix_from_disk(pretrain_data_cfg, data_dir, split="val")
    )
    val_dataloader = create_cycling_pretrain_dataloader(
        val_text_factory,
        tokenizer,
        config.training.pre_training.max_len
    )

    # Создание модели
    logger.info("Инициализация модели GPT")
    model = Transformer(
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

    # Перенос модели на устройство
    model = model.to(device)

    # Логируем параметры модели
    log_model_params(model, logger)
    
    # Логируем память в начале
    log_memory(logger, tag="start")

    optimizer = build_optimizer(
        model,
        config.training.pre_training.learning_rate,
        config.training.pre_training.weight_decay
    )

    # Поиск последнего чекпоинта
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    latest_step = find_latest_checkpoint(checkpoints_dir)

    # Загрузка чекпоинта
    if latest_step is not None:
        load_checkpoint(checkpoint_dir=checkpoints_dir, step=latest_step, model=model, optimizer=optimizer)
        start_step = latest_step + 1
        logger.info(f"Восстановлено обучение с шага {start_step}")
    else:
        start_step = 0

    pretrain_config = config.training.pre_training
    grad_accum_steps = pretrain_config.gradient_accumulation_steps

    # Цикл обучения. Один "step" = один шаг оптимизатора =
    # grad_accum_steps накопленных микро-батчей (эффективный batch size = batch_size * grad_accum_steps)
    for step in range(start_step, pretrain_config.max_steps):
        start_time = time.time()
        lr = get_lr(
            step=step,
            warmup_steps=pretrain_config.warmup_steps,
            max_steps=pretrain_config.max_steps,
            learning_rate=pretrain_config.learning_rate,
            min_learning_rate=pretrain_config.min_learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        accumulated_loss = 0.0

        for _ in range(grad_accum_steps):
            pretrain_batch = collate_pretrain_batch(pretrain_dataloader, pretrain_config.batch_size)
            pretrain_batch = pretrain_batch.to(device)

            loss = train_step(model, pretrain_batch, grad_accum_steps)
            accumulated_loss += loss

            del pretrain_batch

        avg_loss = accumulated_loss / grad_accum_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=pretrain_config.grad_clip_norm)
        optimizer.step()

        elapsed = time.time() - start_time

        log_step(logger, step, avg_loss, lr, grad_norm.item())


        if step % config.monitoring.speed_monitoring.interval_steps == 0:
            log_speed(logger, step, pretrain_config.batch_size, pretrain_config.max_len, elapsed)

        if step % config.monitoring.memory_monitoring.interval_steps == 0:
            log_memory(logger, step, tag="training")

        if step % config.monitoring.gradient_monitoring.interval_steps == 0:
            log_gradients(model, step, logger)


        if step % pretrain_config.checkpoint_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, avg_loss, checkpoints_dir)
            logger.info(f"Чекпоинт сохранён на шаге {step}")

        if step % pretrain_config.eval_interval == 0 and step > 0:
            val_batch = collate_pretrain_batch(val_dataloader, pretrain_config.batch_size)
            val_batch = val_batch.to(device)
            val_loss = eval_step(model, val_batch)
            logger.info(f"step {step}, val_loss={val_loss:.4f}")
            mlflow.log_metric("val_loss", val_loss, step=step)

    # Остановка MLflow
    mlflow.end_run()


if __name__ == "__main__":
    main()
