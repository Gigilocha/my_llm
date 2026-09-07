import torch
import torch.nn as nn
import torch.nn.functional as F

# Функция шага тренировки.
def train_step(
    model: nn.Module, 
    batch: torch.Tensor, 
    grad_accum_steps: int = 1,
    ignore_index: int = -100
) -> float:
    # Сдвиг для авторегрессионной задачи (Next Token Prediction)
    inputs = batch[:, :-1]  # Данные для предсказания
    labels = batch[:, 1:]   # Метки

    device_type = "cuda" if batch.is_cuda else "cpu"
    
    # autocast для matmul/attention в bfloat16
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=batch.is_cuda):
        # Прямой проход по модели
        logits = model(inputs)

        # Выпрямление данных для CrossEntropyLoss
        # Shape: [batch_size * seq_len, vocab_size] vs [batch_size * seq_len]
        logits = logits.view(-1, logits.size(-1))
        labels = labels.reshape(-1)

        # Вычисление ошибки с учетом возможных паддингов
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)

    # Масштабирование лосса для корректного усреднения градиентов при накоплении
    scaled_loss = loss / grad_accum_steps
    scaled_loss.backward()

    # Возвращаем немасштабированное числовое значение loss (для логирования)
    return loss.item()