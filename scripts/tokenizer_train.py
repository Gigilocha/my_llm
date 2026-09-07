from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import build_pretrain_mix_from_disk, extract_texts  # ← локальная версия
from src.tokenizer.tokenizer import train_tokenizer, save_tokenizer

from itertools import islice


def main():
    # Конфиг
    config = get_config()
    # Логгер
    logger = setup_logger(__name__)

    # Директория данных
    data_dir = PROJECT_ROOT / config.env.data_dir

    # Подготовка данных (из локального кэша)
    logger.info("Начинаем сборку pretrain-микса из кэша...")
    mixed = build_pretrain_mix_from_disk(
        config.data.pre_training_data,
        data_dir,
        stage="pretrain",
        split="train"  # берём train сплит
    )

    # Преобразование данных в текст
    text = extract_texts(mixed)
    text = islice(text, config.tokenizer.train_sample_size)

    # Обучение токенизатора
    logger.info("Начинаем обучение токенизатора...")
    tokenizer = train_tokenizer(text, config.tokenizer)

    # Сохранение токенизатора
    save_dir = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    save_tokenizer(tokenizer, save_dir)
    logger.info(f"Токенизатор сохранён в {save_dir}")


if __name__ == "__main__":
    main()