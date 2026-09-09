from transformers import PreTrainedTokenizerFast

from src.tokenizer.tokenizer import encode


"""
Формат одного SFT-примера (спецтокены — из tokenizer_config.yaml):

<|bos|>[{system}\n]<|user_start|>{instruction}[\n{input}]<|user_end|><|assistant_start|>{output}<|assistant_end|><|eos|>

Промпт (всё ДО и ВКЛЮЧАЯ <|assistant_start|>) и ответ токенизируются ОТДЕЛЬНО —
чтобы точно знать границу между ними. В labels промпт помечается IGNORE_INDEX
(-100, значение, которое F.cross_entropy/nn.CrossEntropyLoss игнорирует по
умолчанию) — без этого модель училась бы предсказывать ЧУЖОЙ instruction
(текст пользователя) так же, как собственный ответ, а нас интересует только
"научить генерировать ответ", не "научить писать вопросы пользователя".

Отдельного спецтокена под system нет (только user_start/end, assistant_start/end,
tool_call_*/tool_result_*) — system, если есть, идёт простым текстом в начале,
без обёртки.
"""

IGNORE_INDEX = -100


def format_sft_texts(example: dict, special_tokens: dict) -> tuple[str, str]:
    system_prefix = f"{example['system']}\n" if example.get("system") else ""

    user_content = example["instruction"]
    if example.get("input"):
        user_content += "\n" + example["input"]

    prompt_text = (
        f"{system_prefix}"
        f"{special_tokens['user_start']}{user_content}{special_tokens['user_end']}"
        f"{special_tokens['assistant_start']}"
    )
    response_text = f"{example['output']}{special_tokens['assistant_end']}"

    return prompt_text, response_text


# Токенизирует пример с масками лосса. Возвращает None, если пример не влезает
# даже после обрезки (промпт сам по себе длиннее max_len) — такие пропускаем,
# а не обрубаем ответ на середине (это учило бы модель обрывать генерацию
# произвольно, что нам совершенно не нужно)
def tokenize_sft_example(
    tokenizer: PreTrainedTokenizerFast,
    example: dict,
    special_tokens: dict,
    max_len: int,
) -> tuple[list[int], list[int]] | None:
    prompt_text, response_text = format_sft_texts(example, special_tokens)

    prompt_ids = [tokenizer.bos_token_id] + encode(tokenizer, prompt_text, add_special_tokens=False)
    response_ids = encode(tokenizer, response_text, add_special_tokens=False) + [tokenizer.eos_token_id]

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids

    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        # Обрезаем С НАЧАЛА промпта, не с конца ответа
        if overflow >= len(prompt_ids):
            return None
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    return input_ids, labels


# Поток (input_ids, labels) пар из потока сырых SFT-примеров
def sft_examples_to_tokenized(examples, tokenizer, special_tokens, max_len):
    for example in examples:
        result = tokenize_sft_example(tokenizer, example, special_tokens, max_len)
        if result is not None:
            yield result
