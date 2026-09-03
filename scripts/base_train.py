from pathlib import Path
from transformers import PreTrainedTokenizerFast
import torch
import mlflow

from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.common.mlflow import log_step
from src.data.dataset import build_pretrain_mix_from_disk, extract_texts
from src.data.dataloader import create_pretrain_dataloader, collate_pretrain_batch
from src.model.transformer import Transformer
from src.training.train_step import train_step
from src.training.eval_step import eval_step
from src.training.optimizer import split_params, build_optimizer 
from src.training.lr_schedule import get_lr
from src.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint



def main():
    # Конфиг
    config = get_config()

    # Устройство
    device = config.env.device

    # Логгер
    logger = setup_logger(__name__)

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

    # Сбор тренеровочного микса для Pre-training (train)
    logger.info("Сборка тренировочных данных (train)...")
    train_mix = build_pretrain_mix_from_disk(
        config.data.pre_training_data, 
        data_dir,
        split="train"  
    )
    train_text = extract_texts(train_mix)

    # Сбор оценочного микса для Pre-training (val)
    logger.info("Сборка валидационных данных (val)...")
    val_mix = build_pretrain_mix_from_disk(
        config.data.pre_training_data, 
        data_dir,
        split="val"    
    )
    val_text = extract_texts(val_mix)

    # Инициализация токенизатора
    logger.info("Инициализация токенизатора")
    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    # Подготовка тренеровочноых данных для Pre-training
    train_text = extract_texts(train_mix)
    pretrain_dataloader = create_pretrain_dataloader(
        train_text, 
        tokenizer, 
        config.training.pre_training.max_len
    )

    # Подготовка оценочных данных для Pre-training
    val_text = extract_texts(val_mix)
    val_dataloader = create_pretrain_dataloader(
        val_text, 
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

    optimizer = build_optimizer(
        model, 
        config.training.pre_training.learning_rate, 
        config.training.pre_training.weight_decay  # было pretrain → pre_training
    )

    # Поиск последнего чекпоинта
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    latest_step = find_latest_checkpoint(checkpoints_dir)

    # Загрузка чекпоинта
    if latest_step is not None:
        checkpoint = load_checkpoint(checkpoint_dir=checkpoints_dir, step=latest_step, model=model, optimizer=optimizer)
        start_step = latest_step + 1
    else:
        start_step = 0

    pretrain_config = config.training.pre_training

    # Цикл обучения
    for step in range(start_step, config.training.pre_training.max_steps):
        pretrain_batch = collate_pretrain_batch(
            pretrain_dataloader, 
            pretrain_config.batch_size
        )

        pretrain_batch = pretrain_batch.to(device)
        
        lr = get_lr(
            step=step,
            warmup_steps=pretrain_config.warmup_steps,
            max_steps=pretrain_config.max_steps,
            learning_rate=pretrain_config.learning_rate,
            min_learning_rate=pretrain_config.min_learning_rate
        )
        
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        loss = train_step(model, pretrain_batch)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.pre_training.grad_clip_norm)
        log_step(logger, step, loss, lr, grad_norm.item())
        optimizer.step()
        optimizer.zero_grad()

        if step % config.training.pre_training.checkpoint_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, loss, checkpoints_dir)
            logger.info(f"Чекпоинт сохранён на шаге {step}")

        if step % config.training.pre_training.eval_interval == 0 and step > 0:
            val_batch = collate_pretrain_batch(val_dataloader, config.training.pre_training.batch_size)
            val_batch = val_batch.to(device)
            val_loss = eval_step(model, val_batch)
            logger.info(f"step {step}, val_loss={val_loss:.4f}")
            mlflow.log_metric("val_loss", val_loss, step=step)

        del pretrain_batch

    # Остановка MLflow
    mlflow.end_run()


if __name__ == "__main__":
    main()