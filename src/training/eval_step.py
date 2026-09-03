import torch
import torch.nn as nn
import torch.nn.functional as F


# Функция шага оценки
def eval_step(model: nn.Module, batch: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        inputs = batch[:, :-1]
        labels = batch[:, 1:]

        logits = model(inputs)
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.reshape(-1)

        loss = F.cross_entropy(logits, labels)

    model.train()  # обязательно вернуть обратно — иначе train_step дальше будет работать в eval-режиме
    return loss.item()