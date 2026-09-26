import torch
from transformers import PreTrainedTokenizerFast
from typing import Callable, Iterable, Iterator
from itertools import islice

from src.tokenizer.tokenizer import encode
from src.data.format import IGNORE_INDEX


"""
Единый модуль даталоадеров для ВСЕХ стадий. Внутри — два разных механизма
батчинга, которые исторически были разрезаны по файлам, но не по сути:

1. Pretrain — непрерывный поток текста конкатенируется в один длинный поток
   ТОКЕНОВ и режется на чанки ФИКСИРОВАННОЙ длины (max_len), без padding —
   документы идут подряд, разделённые BOS/EOS.

2. SFT/DPO — примеры дискретные (один пример = один диалог/одна ветка
   preference-пары), с МАСКОЙ лосса (IGNORE_INDEX на промпте) — нужен padding
   до максимальной длины В БАТЧЕ и attention_mask, а не нарезка потока.

Оба реализуют идею "бесконечного" потока (пересоздаём источник, когда данные
кончились, вместо падения на пустом батче) — но по-разному устроены
технически (нарезка буфера токенов vs пересборка потока примеров), поэтому
это не одна функция с параметром, а две родственные — create_cycling_
pretrain_dataloader и create_cycling_tokenized_stream.
"""


# ============================================================================
# Pretrain: поток чанков токенов фиксированной длины
# ============================================================================

# Конечный: заканчивается, когда заканчивается text_stream (один проход по данным)
def create_pretrain_dataloader(
    text_stream: Iterable[str], tokenizer: PreTrainedTokenizerFast, max_seq_len: int
) -> Iterator[list[int]]:
    buffer = []
    for text in text_stream:
        ids = encode(tokenizer, text, True)
        buffer.extend(ids)
        while len(buffer) >= max_seq_len:
            chunk = buffer[:max_seq_len]
            buffer = buffer[max_seq_len:]
            yield chunk
    del buffer


# Бесконечный поток чанков: когда один проход по text_stream заканчивается,
# text_stream пересоздаётся заново через фабрику. Нужен для val, который за
# одно обучение читается много раз (малый объём val быстро истощится иначе)
def create_cycling_pretrain_dataloader(
    text_stream_factory: Callable[[], Iterable[str]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int,
) -> Iterator[list[int]]:
    while True:
        yield from create_pretrain_dataloader(text_stream_factory(), tokenizer, max_seq_len)


def collate_pretrain_batch(dataloader: Iterator[list[int]], batch_size: int) -> torch.Tensor:
    input_ids_list = list(islice(dataloader, batch_size))
    return torch.tensor(input_ids_list)


# ============================================================================
# SFT/DPO: поток дискретных токенизированных примеров + padding-батчинг
# ============================================================================

# Бесконечный поток токенизированных примеров. Общая версия того, что раньше
# было продублировано как cycling_sft_examples в sft_train.py и
# cycling_dpo_examples в rlft_train.py — единственным отличием между ними был
# tokenize_fn (sft_examples_to_tokenized vs dpo_examples_to_tokenized из
# format.py), теперь это параметр:
#
#   create_cycling_tokenized_stream(
#       lambda: build_sft_mix_from_disk(cfg, data_dir, split="train"),
#       lambda examples: sft_examples_to_tokenized(examples, tokenizer, special_tokens, max_len),
#   )
#
# Пересоздаёт examples_factory() заново, когда один проход по данным
# заканчивается — вместо падения на пустом батче. Если за целый проход не
# прошёл токенизацию НИ ОДИН пример (max_len слишком мал относительно длины
# примеров) — без этой проверки цикл крутился бы вечно вхолостую, ничего не
# выдавая и не падая
def create_cycling_tokenized_stream(
    examples_factory: Callable[[], Iterable[dict]],
    tokenize_fn: Callable[[Iterable[dict]], Iterator[tuple[list[int], list[int]]]],
) -> Iterator[tuple[list[int], list[int]]]:
    while True:
        yielded_any = False
        for item in tokenize_fn(examples_factory()):
            yielded_any = True
            yield item
        if not yielded_any:
            raise RuntimeError(
                "Ни один пример не прошёл токенизацию за целый проход по данным — "
                "вероятно max_len слишком мал относительно длины примеров (промпт "
                "со спецтокенами уже превышает max_len). Проверь training_config.yaml"
            )


# Забрать batch_size примеров из потока — общая замена локальному collect_batch
# в sft_train.py и collect_pair_batch в rlft_train.py (там дополнительно
# вызывается дважды: отдельно для chosen и rejected веток)
def collect_examples(dataloader: Iterator, batch_size: int) -> list:
    return list(islice(dataloader, batch_size))


# Паддинг батча SFT/DPO-примеров разной длины до максимальной длины В ЭТОМ
# батче (не до глобального max_len — экономия compute на батчах с короткими
# примерами). attention_mask: 1=реальный токен, 0=паддинг — уходит в
# Transformer.forward(). Общая для SFT и DPO: DPO токенизирует chosen и
# rejected раздельно и вызывает эту функцию дважды на батч (см. rlft_train.py)
def collate_padded_batch(
    examples: list[tuple[list[int], list[int]]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len_in_batch = max(len(ids) for ids, _ in examples)

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for input_ids, labels in examples:
        pad_len = max_len_in_batch - len(input_ids)
        input_ids_batch.append(input_ids + [pad_token_id] * pad_len)
        labels_batch.append(labels + [IGNORE_INDEX] * pad_len)
        attention_mask_batch.append([1] * len(input_ids) + [0] * pad_len)

    return (
        torch.tensor(input_ids_batch, dtype=torch.long),
        torch.tensor(labels_batch, dtype=torch.long),
        torch.tensor(attention_mask_batch, dtype=torch.long),
    )