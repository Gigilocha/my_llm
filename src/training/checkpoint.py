from pathlib import Path
import torch
import torch.nn as nn


# Функция сохраннения контрольной точки обучения
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, loss: float, checkpoint_dir: Path) -> None:

    # Создаем директорию для хранения
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Собираем параметры для сохранения
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    # Сохраняем с номером шага
    torch.save(checkpoint, checkpoint_dir / f"checkpoint_step_{step}.pt")


# Функция загрузки контрольной точки обучения
def load_checkpoint(checkpoint_dir: Path, step: int, model: nn.Module, optimizer: torch.optim.Optimizer):
    
    # Загружаем параметры из файла
    checkpoint = torch.load(checkpoint_dir / f"checkpoint_step_{step}.pt")

    # Восстанавливаем модель и оптмаизатор
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


# Функция проверки чекпоинтов и загрузки последнего чекпоинта
def find_latest_checkpoint(checkpoint_dir: Path) -> int | None:

    # Проверяем, существует ли директория
    if not checkpoint_dir.exists():
        return None

    # Ищем все файлы чекпоинтов
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))

    if not checkpoint_files:
        return None

    # Извлекаем номера шагов из имен файлов
    steps = []
    for file_path in checkpoint_files:
        # Получаем имя файла без расширения
        stem = file_path.stem  # например: "checkpoint_step_1000"

        step = int(stem.split("_")[-1])
        steps.append(step)

    return max(steps)