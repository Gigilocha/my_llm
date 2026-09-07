import torch
import pytest
from src.tokenizer.tokenizer import train_tokenizer
from src.common.config import TokenizerConfig
from src.data.dataloader import create_pretrain_dataloader


# Маленький токенизатор для тестов
def _tiny_tokenizer():
    cfg = TokenizerConfig(
        algorithm="bpe", library="tokenizers", vocab_size=300,
        train_sample_size=100,
        special_tokens={"bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>"},
        split_pattern=r"\S+|\s+",
    )
    corpus = ["hello world", "привет мир", "def foo(): pass"] * 20
    return train_tokenizer(iter(corpus), cfg)


# Проверка точного размера каждого чанка
def test_dataloader_chunk_size_is_exact():
    tokenizer = _tiny_tokenizer()
    texts = iter(["hello world " * 50, "привет мир " * 50, "test test test " * 50])

    chunks = list(create_pretrain_dataloader(texts, tokenizer, max_seq_len=32))

    for chunk in chunks:
        assert len(chunk["input_ids"]) == 32
        assert len(chunk["attention_mask"]) == 32


# Тест сохранение количества токенов
def test_dataloader_preserves_token_count_within_buffer_tolerance():
    tokenizer = _tiny_tokenizer()
    texts = ["hello world " * 50, "привет мир " * 50, "test test test " * 50]

    total_input_tokens = sum(len(tokenizer.encode(t)) + 2 for t in texts)  # +2 за BOS/EOS

    chunks = list(create_pretrain_dataloader(iter(texts), tokenizer, max_seq_len=32))
    total_output_tokens = sum(len(c["input_ids"]) for c in chunks)

    assert total_output_tokens <= total_input_tokens
    assert total_input_tokens - total_output_tokens < 32  # остаток в буфере меньше одного чанка
