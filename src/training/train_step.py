import torch
import torch.nn as nn
import torch.functional as F


# Функция шага тренеровки
def train_step(model: nn.Module, batch: torch.Tensor) -> float:
    inputs = batch[:, :-1] # Получение данных для последующего предсказания
    labels = batch[:, 1:] # Получение меток для предсказания

    # Прямой проход по модели
    logits = model(inputs) 

    # Выпрямление данных 
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.reshape(-1)

    # Вычисление ошибки
    loss = F.cross_entropy(logits, labels)

    # Обратный проход
    loss.backward()

    return loss.item()











    