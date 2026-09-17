from transformers import PreTrainedTokenizerFast

from src.data.sft_format import _format_turn, IGNORE_INDEX
from src.tokenizer.tokenizer import encode


"""
Токенизация preference-пары для DPO. Каждый пример даёт ДВЕ последовательности
с общим промптом: (промпт + chosen) и (промпт + rejected).

Формат промпта и ответа — ТОТ ЖЕ, что в SFT (переиспользуем _format_turn):
модель уже обучена именно на нём, менять разметку между SFT и DPO нельзя,
иначе DPO будет тянуть модель на незнакомом ей формате.

labels используются не для cross-entropy, а чтобы знать, по каким позициям
суммировать log-вероятность ответа: промпт помечен IGNORE_INDEX и в сумму
не входит — сравнивать нужно вероятность ОТВЕТА, а не общего для обеих
веток промпта (он бы просто сократился, но зря тратил бы вычисления и
искажал нормировку по длине).
"""


def _tokenize_side(
    tokenizer: PreTrainedTokenizerFast,
    messages: list[dict],
    response: str,
    special_tokens: dict,
    max_len: int,
) -> tuple[list[int], list[int]] | None:
    input_ids = [tokenizer.bos_token_id]
    labels = [IGNORE_INDEX]

    for message in messages:
        ids = encode(tokenizer, _format_turn(message["role"], message["content"], special_tokens), add_special_tokens=False)
        input_ids.extend(ids)
        labels.extend([IGNORE_INDEX] * len(ids))

    response_ids = encode(tokenizer, _format_turn("assistant", response, special_tokens), add_special_tokens=False)
    input_ids.extend(response_ids)
    labels.extend(response_ids)

    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)

    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        if all(l == IGNORE_INDEX for l in labels):
            return None

    return input_ids, labels


# Возвращает ((chosen_ids, chosen_labels), (rejected_ids, rejected_labels))
# или None, если хотя бы одна из веток не влезает в max_len — обе ветки должны
# быть валидны, иначе пара несравнима
def tokenize_dpo_example(
    tokenizer: PreTrainedTokenizerFast,
    example: dict,
    special_tokens: dict,
    max_len: int,
):
    messages = example["messages"]
    chosen = _tokenize_side(tokenizer, messages, example["chosen"], special_tokens, max_len)
    rejected = _tokenize_side(tokenizer, messages, example["rejected"], special_tokens, max_len)

    if chosen is None or rejected is None:
        return None
    return chosen, rejected


def dpo_examples_to_tokenized(examples, tokenizer, special_tokens, max_len):
    for example in examples:
        result = tokenize_dpo_example(tokenizer, example, special_tokens, max_len)
        if result is not None:
            yield result
