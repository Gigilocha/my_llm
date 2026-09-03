import mlflow
import logging
import torch
import psutil
import time
from pathlib import Path
from typing import Optional


# Логирует шаг обучения.
def log_step(logger: logging.Logger, step: int, loss: float, lr: float, grad_norm: float) -> None:
    logger.info(f"step {step}, loss={loss:.4f}, lr={lr:.6f}, grad_norm={grad_norm:.4f}")
    mlflow.log_metrics({
        "train_loss": loss,
        "learning_rate": lr,
        "grad_norm": grad_norm,
    }, step=step)


# Логирует валидационный loss.
def log_eval(logger: logging.Logger, step: int, val_loss: float) -> None:
    logger.info(f"step {step}, val_loss={val_loss:.4f}")
    mlflow.log_metric("val_loss", val_loss, step=step)


# Логирует статистику градиентов (норма, среднее, макс/мин).
def log_gradients(model, step: int, logger: Optional[logging.Logger] = None) -> None:
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            # Опционально: логировать каждый слой
            # mlflow.log_metric(f"grad_norm_{name}", grad_norm, step=step)
    
    if grad_norms:
        avg_grad = sum(grad_norms) / len(grad_norms)
        max_grad = max(grad_norms)
        min_grad = min(grad_norms)
        
        mlflow.log_metrics({
            "grad_avg": avg_grad,
            "grad_max": max_grad,
            "grad_min": min_grad,
        }, step=step)
        
        if logger:
            logger.debug(f"gradients: avg={avg_grad:.4f}, max={max_grad:.4f}, min={min_grad:.4f}")


# Логирует использование VRAM и RAM.
def log_memory(logger: logging.Logger, step: Optional[int] = None, tag: str = "") -> None:
    metrics = {}
    
    # VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        
        metrics.update({
            "vram_allocated_gb": allocated,
            "vram_reserved_gb": reserved,
            "vram_max_gb": max_allocated,
        })
        
        logger.info(f"[{tag}] VRAM: {allocated:.2f}/{reserved:.2f}/{max_allocated:.2f} GB")
    
    # RAM
    ram = psutil.virtual_memory()
    metrics.update({
        "ram_used_gb": ram.used / 1024**3,
        "ram_total_gb": ram.total / 1024**3,
        "ram_percent": ram.percent,
    })
    
    logger.info(f"[{tag}] RAM: {ram.used / 1024**3:.2f}/{ram.total / 1024**3:.2f} GB ({ram.percent}%)")
    
    # Логируем в MLflow
    if step is not None:
        mlflow.log_metrics(metrics, step=step)


# Логирует скорость обучения (токены/сек, шаги/сек).
def log_speed(
    logger: logging.Logger,
    step: int,
    batch_size: int,
    seq_len: int,
    elapsed_time: float,
) -> None:
    # Токены в батче: batch_size * seq_len
    tokens_per_step = batch_size * seq_len
    tokens_per_sec = tokens_per_step / elapsed_time
    steps_per_sec = 1.0 / elapsed_time
    
    mlflow.log_metrics({
        "tokens_per_sec": tokens_per_sec,
        "steps_per_sec": steps_per_sec,
        "batch_time_sec": elapsed_time,
    }, step=step)
    
    logger.info(f"speed: {tokens_per_sec:.0f} tok/s, {steps_per_sec:.2f} step/s")


# Логирует количество параметров модели.
def log_model_params(model, logger: logging.Logger) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 Model params: {total_params:,} ({total_params / 1e6:.2f}M)")
    logger.info(f"   Trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    
    mlflow.log_params({
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": total_params * 4 / 1024**2,  # float32
    })


# Сохраняет артефакты (конфиги, логи) в MLflow.
def log_artifacts(experiment_dir: Path) -> None:
    if experiment_dir.exists():
        # Конфиги
        configs_dir = experiment_dir / "configs"
        if configs_dir.exists():
            mlflow.log_artifacts(str(configs_dir), artifact_path="configs")
        
        # Логи
        logs_dir = experiment_dir / "logs"
        if logs_dir.exists():
            mlflow.log_artifacts(str(logs_dir), artifact_path="logs")


# Сохраняет модель в MLflow.
def log_model(model, artifact_path: str = "model") -> None:
    try:
        import torch
        mlflow.pytorch.log_model(model, artifact_path)
    except Exception as e:
        logging.warning(f"Failed to log model to MLflow: {e}")