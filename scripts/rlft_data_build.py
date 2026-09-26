from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import cache_dpo_data


# Докачка preference-данных для DPO отдельно от запуска обучения —
# по аналогии с data_build.py (pretrain) и sft_data_build.py
def main():
    config = get_config()
    logger = setup_logger(__name__)

    cfg = config.data.rlft_data
    data_dir = PROJECT_ROOT / config.env.data_dir

    if not cfg.sources:
        logger.info("rlft_data.sources пуст — добавь источники preference-пар в configs/data_config.yaml")
        return

    logger.info(f"Кеширование DPO: {len(cfg.sources)} источников, cache_docs_per_source={cfg.cache_docs_per_source}")
    cache_dpo_data(cfg, data_dir)
    logger.info("Кеширование DPO завершено")


if __name__ == "__main__":
    main()
