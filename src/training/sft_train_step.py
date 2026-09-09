import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.sft_format import IGNORE_INDEX


# Аналог train_step для SFT: та же логика (сдвиг на 1, autocast, деление на
# grad_accum_steps перед backward), но с двумя отличиями:
# 1) labels приходят готовыми (с -100 на позициях промпта), не вычисляются
#    сдвигом input_ids — сдвигаем и labels, и attention_mask на 1 синхронно
# 2) ignore_index=-100 в cross_entropy — токены промпта и паддинга не участвуют
#    в лоссе вообще (градиент через них не идёт)
def sft_train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    grad_accum_steps: int = 1,
) -> float:
    inputs = input_ids[:, :-1]
    targets = labels[:, 1:]
    mask = attention_mask[:, :-1]

    device_type = "cuda" if input_ids.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=input_ids.is_cuda):
        logits = model(inputs, attention_mask=mask)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)

    (loss / grad_accum_steps).backward()
    return loss.item()


@torch.no_grad()
def sft_eval_step(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> float:
    model.eval()
    inputs = input_ids[:, :-1]
    targets = labels[:, 1:]
    mask = attention_mask[:, :-1]

    device_type = "cuda" if input_ids.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=input_ids.is_cuda):
        logits = model(inputs, attention_mask=mask)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX)

    model.train()
    return loss.item()
