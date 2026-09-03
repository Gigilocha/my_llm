import mlflow
import logging


# Шаг логирования метрик в MLflow
def log_step(logger: logging.Logger, step: int, loss: float, lr: float, grad_norm: float) -> None:
    logger.info(f"step {step}, loss={loss:.4f}, lr={lr:.6f}, grad_norm={grad_norm:.4f}")
    mlflow.log_metrics({"loss": loss, "lr": lr, "grad_norm": grad_norm}, step=step)

