from transformers import PreTrainedTokenizerFast
from typing import Iterator

from src.tokenizer import encode


# Бесконечный поток токенов
def create_pretrain_dataloader(
        text_stream: Iterator[str], 
        tokenizer: PreTrainedTokenizerFast, 
        max_seq_len: int,
    ) -> Iterator[dict]:

    # Буфер накапливания 
    buffer = []

    # Токенизация текста
    for text in text_stream:
        ids = encode(tokenizer, text, True)

        # Добавление в буффер
        buffer.extend(ids)

        # Отдача чанков
        while len(buffer) >= max_seq_len:
            chunk = buffer[:max_seq_len]
            buffer = buffer[max_seq_len:]
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
            }



