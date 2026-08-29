from pathlib import Path
from transformers import PreTrainedTokenizerFast
from src.common.config import TokenizerConfig
from src.tokenizer.tokenizer import train_tokenizer, save_tokenizer, encode, decode


def _tiny_config() -> TokenizerConfig:
    return TokenizerConfig(
        algorithm="bpe",
        library="tokenizers",
        vocab_size=300,
        train_sample_size=100,
        special_tokens={"bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>"},
        split_pattern=r"\S+|\s+",
    )


TINY_CORPUS = [
    "Привет, как дела?",
    "Hello, how are you?",
    "def foo(): return 1",
] * 20  # повторяем, чтобы BPE было из чего строить merges


def test_train_tokenizer_returns_fast_tokenizer():
    tokenizer = train_tokenizer(iter(TINY_CORPUS), _tiny_config())
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert tokenizer.bos_token == "<|bos|>"
    assert tokenizer.eos_token == "<|eos|>"


def test_encode_decode_roundtrip():
    tokenizer = train_tokenizer(iter(TINY_CORPUS), _tiny_config())
    text = "Привет, how are you?"
    ids = encode(tokenizer, text)
    decoded = decode(tokenizer, ids)
    assert decoded == text


def test_save_tokenizer_creates_files(tmp_path):
    tokenizer = train_tokenizer(iter(TINY_CORPUS), _tiny_config())
    save_dir = tmp_path / "tokenizer"
    save_tokenizer(tokenizer, save_dir)
    assert (save_dir / "tokenizer.json").exists()