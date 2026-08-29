from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import build_pretrain_mix, extract_texts
from src.tokenizer.tokenizer import train_tokenizer, save_tokenizer

from itertools import islice


def main():
    # Конфиг
    config = get_config()
    # Логгер
    logger = setup_logger(__name__)

    # Подготовка данных
    logger.info("Начинаем сборку pretrain-микса")
    mixed = build_pretrain_mix(config.data.pre_training_data)

    # Преобразование данных в текст
    text = extract_texts(mixed)
    text = islice(text, config.tokenizer.train_sample_size) 

    # Обучение токенизатора
    logger.info("Начинаем обучение токенизатора")
    tokenizer = train_tokenizer(text, config.tokenizer)

    # Сохранение токенизатора
    save_dir = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    save_tokenizer(tokenizer, save_dir)
    logger.info(f"Токенизатор сохранён в {save_dir}")


if __name__ == "__main__":
    main()