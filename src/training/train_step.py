import torch
import torch.nn as nn
import torch.nn.functional as F


# Функция шага тренеровки.
# grad_accum_steps > 1: loss делится перед backward, чтобы градиенты нескольких
# микро-батчей усреднялись, а не суммировались — иначе эффективный lr незаметно
# вырастает в grad_accum_steps раз. Возвращается немасштабированный loss (для логов).
def train_step(model: nn.Module, batch: torch.Tensor, grad_accum_steps: int = 1) -> float:
    inputs = batch[:, :-1] # Получение данных для последующего предсказания
    labels = batch[:, 1:] # Получение меток для предсказания

    # Прямой проход по модели
    logits = model(inputs) 

    # Выпрямление данных 
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.reshape(-1)

    # Вычисление ошибки
    loss = F.cross_entropy(logits, labels)

    # Обратное распространение (усредняем вклад микро-батча в общий градиент шага)
    (loss / grad_accum_steps).backward()

    return loss.item()











    