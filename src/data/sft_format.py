from transformers import PreTrainedTokenizerFast

from src.tokenizer.tokenizer import encode


"""
Работаем с единым внутренним форматом {"messages": [{"role": ..., "content": ...}, ...]}
независимо от того, из какого источника пришёл пример (messages-датасет или
instruction/output — sft_dataset.py нормализует оба в это представление).

Проходим по messages по очереди: system/user — часть промпта (IGNORE_INDEX
в labels), assistant — часть ответа (реальные labels, участвует в лоссе).
Поддерживает multi-turn "из коробки" — просто больше пар user/assistant подряд.

"think"-блоки (reasoning-трейсы вида <think>...</think> перед финальным
ответом) отдельной обработки не требуют — это просто текст ВНУТРИ content
хода assistant, участвует в лоссе наравне с остальным ответом (так и должно
быть: reasoning — часть того, что модель должна научиться генерировать, а не
служебная разметка). Отдельных спецтокенов под think нет — токенизируются как
обычный текст, не как атомарный токен; если станет важна эффективность токенов,
можно добавить <think>/<\\think> в спецтокены токенизатора отдельно.
"""

IGNORE_INDEX = -100


# Формат ходов (общий для tokenize_sft_example и format_prompt_for_generation):
# system  -> "{content}\n"
# user    -> "<|user_start|>{content}<|user_end|><|assistant_start|>"
# assistant -> "{content}<|assistant_end|>"
def _format_turn(role: str, content: str, special_tokens: dict) -> str:
    if role == "system":
        return f"{content}\n"
    if role == "user":
        return f"{special_tokens['user_start']}{content}{special_tokens['user_end']}{special_tokens['assistant_start']}"
    if role == "assistant":
        return f"{content}{special_tokens['assistant_end']}"
    raise ValueError(f"Неизвестная роль: {role}")


# Токенизирует полный многоходовый диалог с масками лосса.
# Возвращает None, если после обрезки по max_len не осталось ни одного видимого
# (не -100) токена — такой пример бесполезен для обучения (модели нечему учиться)
def tokenize_sft_example(
    tokenizer: PreTrainedTokenizerFast,
    example: dict,
    special_tokens: dict,
    max_len: int,
) -> tuple[list[int], list[int]] | None:
    messages = example["messages"]

    input_ids = [tokenizer.bos_token_id]
    labels = [IGNORE_INDEX]

    for message in messages:
        text = _format_turn(message["role"], message["content"], special_tokens)
        ids = encode(tokenizer, text, add_special_tokens=False)
        input_ids.extend(ids)
        # Виден в лоссе только ответ ассистента — всё остальное (system/user,
        # включая user_start/user_end/assistant_start) размечается IGNORE_INDEX
        if message["role"] == "assistant":
            labels.extend(ids)
        else:
            labels.extend([IGNORE_INDEX] * len(ids))

    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)

    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        # Обрезаем С НАЧАЛА диалога (самые старые ходы) — не с конца, чтобы не
        # обрубать последний ответ ассистента на середине
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]
        if all(l == IGNORE_INDEX for l in labels):
            return None

    return input_ids, labels


# Строит текст промпта для ГЕНЕРАЦИИ (инференс) — все ходы, включая прошлые
# ответы ассистента (история диалога), заканчивая на <|assistant_start|>
# после последнего user-хода. BOS не добавляется здесь — его добавляет сама
# generate()/GenerationEngine.generate(), как и при обучении
def format_prompt_for_generation(messages: list[dict], special_tokens: dict) -> str:
    return "".join(_format_turn(m["role"], m["content"], special_tokens) for m in messages)


# Поток (input_ids, labels) пар из потока сырых SFT-примеров
def sft_examples_to_tokenized(examples, tokenizer, special_tokens, max_len):
    for example in examples:
        result = tokenize_sft_example(tokenizer, example, special_tokens, max_len)
        if result is not None:
            yield result
