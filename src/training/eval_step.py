import torch
import torch.nn as nn
import torch.nn.functional as F


# Функция шага оценки (валидации)
@torch.no_grad()
def eval_step(
    model: nn.Module, 
    batch: torch.Tensor, 
    ignore_index: int = -100
) -> float:
    # Переводим модель в режим оценки
    was_training = model.training
    model.eval()

    inputs = batch[:, :-1]
    labels = batch[:, 1:]

    device_type = "cuda" if batch.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=batch.is_cuda):
        logits = model(inputs)

        # Выпрямление данных для CrossEntropyLoss
        logits = logits.view(-1, logits.size(-1))
        labels = labels.reshape(-1)

        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)

    # Восстанавливаем исходный режим модели (train или eval)
    if was_training:
        model.train()

    return loss.item()