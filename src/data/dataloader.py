import torch
from transformers import PreTrainedTokenizerFast
from typing import Callable, Iterable, Iterator
from itertools import islice

from src.tokenizer.tokenizer import encode


# Поток чанков токенов фиксированной длины из потока текстов.
# Конечный: заканчивается, когда заканчивается text_stream (один проход по данным).
def create_pretrain_dataloader(text_stream: Iterable[str], tokenizer: PreTrainedTokenizerFast, max_seq_len: int) -> Iterator[list[int]]:
    buffer = []
    for text in text_stream:
        ids = encode(tokenizer, text, True)
        buffer.extend(ids)
        while len(buffer) >= max_seq_len:
            chunk = buffer[:max_seq_len]
            buffer = buffer[max_seq_len:]
            yield chunk
    del buffer


# Бесконечный поток чанков токенов: когда один проход по text_stream заканчивается,
# text_stream пересоздаётся заново через фабрику. Нужен для val, который за одно
# обучение читается много раз (при малом объёме val-данных обычный проход быстро истощится)
def create_cycling_pretrain_dataloader(
    text_stream_factory: Callable[[], Iterable[str]],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_len: int
) -> Iterator[list[int]]:
    while True:
        yield from create_pretrain_dataloader(text_stream_factory(), tokenizer, max_seq_len)


# Функция создания батча
def collate_pretrain_batch(dataloader: Iterator[list[int]], batch_size: int) -> torch.Tensor:
    input_ids_list = list(islice(dataloader, batch_size))
    return torch.tensor(input_ids_list)
