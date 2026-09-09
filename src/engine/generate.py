import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from src.tokenizer.tokenizer import encode, decode


"""
Наивная авторегрессионная генерация: KV-кеша нет, на каждом новом токене
прогоняем ВСЮ последовательность целиком через модель заново (O(n^2) по длине).
Для интерактивного инференса это будет неприемлемо медленно — KV-кеш это
отдельная задача (собственно то, для чего заводили src/engine изначально).
Но для разовой оценки чекпоинта — сгенерировать десяток коротких примеров
и почитать глазами — этого достаточно, не хотелось тащить сложность кеша
только ради этого.
"""


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    device: str,
    max_position_embeddings: int,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
    repetition_penalty: float = 1.3,
) -> str:
    was_training = model.training
    model.eval()

    # BOS в начале, но БЕЗ EOS — encode(..., add_special_tokens=True) добавил бы
    # EOS в конец промпта, что означало бы для модели "последовательность уже
    # закончена" ровно в момент, когда мы просим её продолжать
    prompt_ids = encode(tokenizer, prompt, add_special_tokens=False)
    ids = [tokenizer.bos_token_id] + prompt_ids

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Модель не видела позиций дальше max_position_embeddings при обучении —
        # если промпт+генерация вылезли за окно, оставляем только последний кусок
        context = input_ids[:, -max_position_embeddings:]

        logits = model(context)
        next_token_logits = logits[0, -1, :]

        # Repetition penalty (Keskar et al., CTRL): штрафуем токены, которые уже
        # встречались в сгенерированной последовательности — понижаем логит, если
        # он положительный, повышаем "штраф" (делаем более отрицательным), если
        # уже отрицательный. Без этого модель на недообученных данных (как у нас
        # сейчас) легко проваливается в буквальные повторы — например, генерирует
        # два идентичных def подряд, потому что "продолжить тем же самым" для неё
        # локально самый вероятный токен
        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                if next_token_logits[token_id] > 0:
                    next_token_logits[token_id] /= repetition_penalty
                else:
                    next_token_logits[token_id] *= repetition_penalty

        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        else:
            # temperature=0 — жадная генерация (argmax), без сэмплирования
            next_token = next_token_logits.argmax().unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            continue

        if top_k is not None:
            top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            threshold = top_k_values[-1]
            next_token_logits = torch.where(
                next_token_logits < threshold,
                torch.full_like(next_token_logits, float("-inf")),
                next_token_logits,
            )

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Отсекаем хвост, как только накопленная вероятность превысила top_p —
            # первый токен (даже если сам по себе даёт больше top_p) оставляем всегда
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_mask] = float("-inf")
            next_token_logits = torch.full_like(next_token_logits, float("-inf"))
            next_token_logits[sorted_indices] = sorted_logits

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    if was_training:
        model.train()

    generated_ids = input_ids[0].tolist()
    return decode(tokenizer, generated_ids)
