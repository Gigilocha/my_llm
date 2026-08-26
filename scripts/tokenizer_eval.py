from pathlib import Path
from transformers import PreTrainedTokenizerFast

from src.common.config import PROJECT_ROOT, get_config
from src.common.logger import setup_logger


# Корпус текстов
TEST_CORPUS = {
    "rus": [
        "Привет, как у тебя дела сегодня?",
        "Машинное обучение — это интересная область исследований.",
        "Токенизатор должен хорошо работать с кириллицей.",
    ],
    "en": [
        "Hello, how are you doing today?",
        "Machine learning is a fascinating field of research.",
        "The tokenizer should handle English text efficiently.",
    ],
    "code": [
        "def train_tokenizer(corpus, cfg):\n    return tokenizer",
        "for i in range(10):\n    print(i ** 2)",
        "class Model(nn.Module):\n    def __init__(self):\n        super().__init__()",
    ],
}


# Функция подсчёта compression ratio (Сколько токенов приходится на условную единицу текста)
def compression_ratio(text: str, tokenizer: PreTrainedTokenizerFast) -> float:
    # Количество символов в тексте
    text_length = len(text)

    # Количество токенов
    ids = tokenizer.encode(text)
    num_tokens = len(ids)

    # Коэффициент сжатия
    if num_tokens == 0:
        return 0.0
    return text_length / num_tokens


# Функция проверки round-trip (Декодирование без потерь)
def check_roundtrip(text: str, tokenizer: PreTrainedTokenizerFast) -> bool:
    # Сохранение оригинального текста
    original_text = text

    # Кодируем текст
    encode_text = tokenizer.encode(text)

    # Декодируем текст обратно
    decode_text = tokenizer.decode(encode_text)

    if original_text == decode_text:
        return True
    else:
        return False


def main():
    logger = setup_logger(__name__)
    config = get_config()
    tokenizer_dir = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    for language, texts in TEST_CORPUS.items():
        logger.info(f"--- {language} ---")
        for text in texts:
            ratio = compression_ratio(text, tokenizer)
            ok = check_roundtrip(text, tokenizer)
            logger.info(f"ratio={ratio:.2f} chars/token, roundtrip_ok={ok} | {text[:40]}")


if __name__ == "__main__":
    main()