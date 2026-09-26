from transformers import PreTrainedTokenizerFast

from src.tokenizer.tokenizer import encode


"""
Единый модуль форматирования диалогов для SFT и DPO. Раньше dpo_format.py
уже импортировал _format_turn/IGNORE_INDEX из sft_format.py — то есть
зависимость была односторонняя и файлы были разрезаны искусственно, а не
по сути. DPO должен использовать ТОТ ЖЕ формат хода диалога, что и SFT
(иначе на DPO-этапе модель встретит незнакомую ей разметку) — это одна
логика форматирования на две стадии, а не две логики.

Работаем с единым внутренним форматом {"messages": [{"role": ..., "content":
...}, ...]} независимо от того, из какого источника пришёл пример
(dataset.py уже нормализует и SFT, и DPO-промпт к этому представлению).

Проходим по messages по очереди: system/user — часть промпта (IGNORE_INDEX
в labels), assistant — часть ответа (реальные labels, участвует в лоссе).
Поддерживает multi-turn "из коробки" — просто больше пар user/assistant подряд.

"think"-блоки (reasoning-трейсы вида <think>...</think> перед финальным
ответом) отдельной обработки не требуют — это просто текст ВНУТРИ content
хода assistant, участвует в лоссе наравне с остальным ответом (так и должно
быть: reasoning — часть того, что модель должна научиться генерировать, а не
служебная разметка). Отдельных спецтокенов под think нет — токенизируются
как обычный текст, не как атомарный токен.
"""

IGNORE_INDEX = -100


# Формат ходов (общий для SFT-токенизации, DPO-токенизации и генерации):
# system    -> "{content}\n"
# user      -> "<|user_start|>{content}<|user_end|><|assistant_start|>"
# assistant -> "{content}<|assistant_end|>"
def _format_turn(role: str, content: str, special_tokens: dict) -> str:
    if role == "system":
        return f"{content}\n"
    if role == "user":
        return f"{special_tokens['user_start']}{content}{special_tokens['user_end']}{special_tokens['assistant_start']}"
    if role == "assistant":
        return f"{content}{special_tokens['assistant_end']}"
    raise ValueError(f"Неизвестная роль: {role}")


# ============================================================================
# SFT
# ============================================================================

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


# ============================================================================
# DPO
# ============================================================================

# Токенизация ОДНОЙ ветки (промпт + один из ответов) — общая для chosen и
# rejected, вызывается дважды на пример из tokenize_dpo_example(). labels
# используются не для cross-entropy, а чтобы знать, по каким позициям
# суммировать log-вероятность ответа: промпт помечен IGNORE_INDEX и в сумму
# не входит — сравнивать нужно вероятность ОТВЕТА, а не общего для обеих
# веток промпта (он бы просто сократился, но зря тратил бы вычисления и
# искажал нормировку по длине)
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