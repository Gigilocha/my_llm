import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.format import IGNORE_INDEX


"""
DPO (Direct Preference Optimization, Rafailov et al.).

Идея: вместо обучения reward-модели и RL-роллаутов (PPO/GRPO) оптимизируем
предпочтения напрямую. На паре (chosen, rejected) максимизируем

    log sigmoid( beta * [ (logp_pol(chosen) - logp_ref(chosen))
                        - (logp_pol(rejected) - logp_ref(rejected)) ] )

Разности с reference-моделью (замороженная копия SFT-модели) удерживают
политику от ухода в вырожденный текст: без них ничто не мешает модели
повышать вероятность chosen, ломая язык в целом.
"""


# Сумма log-вероятностей токенов ОТВЕТА (позиции, где labels != IGNORE_INDEX).
# Возвращает [batch] — по одному числу на последовательность
def sequence_logprob(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    inputs = input_ids[:, :-1]
    targets = labels[:, 1:]
    mask = attention_mask[:, :-1]

    logits = model(inputs, attention_mask=mask)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    valid = targets != IGNORE_INDEX
    # gather не принимает -100 как индекс — подставляем 0 на невалидных
    # позициях и зануляем их вклад маской после gather
    safe_targets = targets.masked_fill(~valid, 0)
    token_logprobs = log_probs.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)

    return (token_logprobs * valid).sum(dim=-1)


# Один шаг DPO. policy_model обучается, ref_model заморожена (no_grad).
# Возвращает (loss, метрики для логов)
def dpo_train_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float,
    grad_accum_steps: int = 1,
) -> tuple[float, dict]:
    device_type = "cuda" if chosen_ids.is_cuda else "cpu"
    autocast_on = chosen_ids.is_cuda

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_on):
        policy_chosen = sequence_logprob(policy_model, chosen_ids, chosen_labels, chosen_mask)
        policy_rejected = sequence_logprob(policy_model, rejected_ids, rejected_labels, rejected_mask)

        with torch.no_grad():
            ref_chosen = sequence_logprob(ref_model, chosen_ids, chosen_labels, chosen_mask)
            ref_rejected = sequence_logprob(ref_model, rejected_ids, rejected_labels, rejected_mask)

    # Логиты DPO считаем в fp32 — разности логарифмов вероятностей длинных
    # последовательностей легко выходят за диапазон стабильности bf16
    policy_diff = policy_chosen.float() - policy_rejected.float()
    ref_diff = ref_chosen.float() - ref_rejected.float()
    dpo_logits = beta * (policy_diff - ref_diff)

    loss = -F.logsigmoid(dpo_logits).mean()
    (loss / grad_accum_steps).backward()

    with torch.no_grad():
        # accuracy: доля пар, где политика уже предпочитает chosen сильнее,
        # чем это делала reference-модель. 0.5 = не научилась ничему,
        # рост к 1.0 = предпочтения усваиваются
        accuracy = (dpo_logits > 0).float().mean().item()
        chosen_reward = beta * (policy_chosen.float() - ref_chosen.float()).mean().item()
        rejected_reward = beta * (policy_rejected.float() - ref_rejected.float()).mean().item()

    metrics = {
        "accuracy": accuracy,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "reward_margin": chosen_reward - rejected_reward,
    }
    return loss.item(), metrics


@torch.no_grad()
def dpo_eval_step(
    policy_model, ref_model,
    chosen_ids, chosen_labels, chosen_mask,
    rejected_ids, rejected_labels, rejected_mask,
    beta: float,
) -> tuple[float, float]:
    policy_model.eval()
    device_type = "cuda" if chosen_ids.is_cuda else "cpu"

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=chosen_ids.is_cuda):
        policy_chosen = sequence_logprob(policy_model, chosen_ids, chosen_labels, chosen_mask)
        policy_rejected = sequence_logprob(policy_model, rejected_ids, rejected_labels, rejected_mask)
        ref_chosen = sequence_logprob(ref_model, chosen_ids, chosen_labels, chosen_mask)
        ref_rejected = sequence_logprob(ref_model, rejected_ids, rejected_labels, rejected_mask)

    dpo_logits = beta * ((policy_chosen.float() - policy_rejected.float()) - (ref_chosen.float() - ref_rejected.float()))
    loss = -F.logsigmoid(dpo_logits).mean().item()
    accuracy = (dpo_logits > 0).float().mean().item()

    policy_model.train()
    return loss, accuracy
