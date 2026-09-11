from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.sft_dataset import cache_sft_data


# Докачка/кеширование SFT-данных отдельно от запуска обучения — та же логика,
# что cache_sft_data() внутри base_sft.py (она всё равно идемпотентна и
# пропустит уже готовые источники), но так можно докачать данные заранее,
# не запуская сразу полный обучающий луп — по аналогии с data_build.py для pretrain
def main():
    config = get_config()
    logger = setup_logger(__name__)

    cfg = config.data.sft_data
    data_dir = PROJECT_ROOT / config.env.data_dir

    logger.info(f"Кеширование SFT: {len(cfg.sources)} источников, cache_docs_per_source={cfg.cache_docs_per_source}")
    cache_sft_data(cfg, data_dir)
    logger.info("Кеширование SFT завершено")


if __name__ == "__main__":
    main()
