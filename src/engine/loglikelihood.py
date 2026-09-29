import torch
import torch.nn.functional as F

from src.tokenizer.tokenizer import encode as tokenizer_encode


"""
Log-вероятность continuation, данного context — общая реализация для
lm_eval_adapter.py (интеграция с lm-evaluation-harness через loglikelihood())
и benchmark.py (прямой multiple-choice скоринг без обвязки harness: для трёх
source в этом прогоне регистрировать кастомные задачи в lm-evaluation-harness
было бы отдельной работой, не пропорциональной самому скорингу).
"""


@torch.no_grad()
def sequence_loglikelihood(
    model,
    tokenizer,
    context: str,
    continuation: str,
    device: str,
    max_position_embeddings: int,
) -> tuple[float, bool]:
    context_ids = tokenizer_encode(tokenizer, context, add_special_tokens=False) if context else []
    continuation_ids = tokenizer_encode(tokenizer, continuation, add_special_tokens=False)

    full_ids = [tokenizer.bos_token_id] + context_ids + continuation_ids
    cont_len = len(continuation_ids)

    # Обрезаем СЛЕВА (старый контекст), если не влезает — continuation
    # должен остаться в контексте целиком, иначе нечего скорить
    if len(full_ids) > max_position_embeddings:
        full_ids = full_ids[-max_position_embeddings:]

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(input_ids)  # [1, seq_len, vocab]

    # Логиты на позиции i предсказывают токен i+1 — continuation начинается
    # с позиции (len(full_ids) - cont_len), значит предсказывающие его
    # логиты лежат на один индекс раньше
    cont_start = len(full_ids) - cont_len
    pred_logits = logits[0, cont_start - 1: -1, :]
    target_ids = torch.tensor(full_ids[cont_start:], dtype=torch.long, device=device)

    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    total_logprob = token_log_probs.sum().item()

    greedy_ids = pred_logits.argmax(dim=-1)
    is_greedy = bool((greedy_ids == target_ids).all().item())

    return total_logprob, is_greedy