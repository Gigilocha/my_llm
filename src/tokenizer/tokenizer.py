from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders, processors, trainers
from transformers import PreTrainedTokenizerFast
from src.common.config import TokenizerConfig
from typing import Iterator
from pathlib import Path


# Обучение токенизатора
def train_tokenizer(corpus: Iterator[str], cfg: TokenizerConfig) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.BPE(unk_token=None)) 
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(cfg.split_pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Преобразование словаря в список
    special_tokens_list = list(cfg.special_tokens.values())

    # Тренер токенизатора
    trainer = trainers.BpeTrainer(
        vocab_size = cfg.vocab_size,
        special_tokens = special_tokens_list,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Обучение токенизатора
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Преобразование токенизатора
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        bos_token=cfg.special_tokens["bos"],
        eos_token=cfg.special_tokens["eos"],
        pad_token=cfg.special_tokens.get("pad"),
        additional_special_tokens=[v for k, v in cfg.special_tokens.items() if k not in ("bos", "eos", "pad")],
    )
    return fast_tokenizer


# Сохранение токенизатора
def save_tokenizer(tokenizer: PreTrainedTokenizerFast, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)


# Функции для Pre-training
# Кодирование текста 
def encode(tokenizer: PreTrainedTokenizerFast, text: str, add_special_tokens: bool = True) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if add_special_tokens:
        return [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id] 
    return ids


# Декодирование текста
def decode(tokenizer: PreTrainedTokenizerFast, ids: list[int]) -> str:
    return tokenizer.decode(ids)


# Функции для Fine-tuning
# Кодирование сообщения
# def encode_messege(message):


# Форматирвоание диалогов в плоскую последовательность
# def render_conversation


# Возвращение маски для спец токенов, дабы модель не генерила не нужное
# def build_loss_mask

