from itertools import islice

from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import build_pretrain_mix_from_disk, extract_texts
from src.tokenizer.tokenizer import train_tokenizer, save_tokenizer


"""
Читает уже ЗАКЕШИРОВАННЫЕ на диск данные (запусти base_data_build.py раньше),
а не собирает отдельный сетевой микс, как было раньше. Причина: калибровка
пропорций (length_fn=len — по символам, токенизатора для измерения по
токенам ещё не существует, он тут как раз и обучается) должна идти по ТОЙ ЖЕ
выборке, что потом увидит реальное обучение — иначе токенизатор может выйти
недообученным на домене (например, коде), который в его собственной
тренировочной выборке был представлен не в тех пропорциях, что в финальном
pretrain-корпусе.
"""


def main():
    config = get_config()
    logger = setup_logger(__name__)

    data_dir = PROJECT_ROOT / config.env.data_dir
    pretrain_cfg = config.data.pre_training_data

    logger.info("Сборка pretrain-микса с диска (калибровка по символам)")
    mixed = build_pretrain_mix_from_disk(pretrain_cfg, data_dir, length_fn=len)

    text = extract_texts(mixed)
    text = islice(text, config.tokenizer.train_sample_size)

    logger.info("Начинаем обучение токенизатора")
    tokenizer = train_tokenizer(text, config.tokenizer)

    save_dir = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    save_tokenizer(tokenizer, save_dir)
    logger.info(f"Токенизатор сохранён в {save_dir}")


if __name__ == "__main__":
    main()