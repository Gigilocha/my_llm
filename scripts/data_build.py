from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import cache_pretrain_data


# Загрузка датасета на диск 
def main():
    config = get_config()
    logger = setup_logger(__name__)

    cfg = config.data.pre_training_data
    data_dir = PROJECT_ROOT / config.env.data_dir

    logger.info(
        f"Кеширование: rus={cfg.rus_cache_docs}, en={cfg.en_cache_docs}, code={cfg.code_cache_docs} документов"
    )
    cache_pretrain_data(cfg, data_dir)
    logger.info("Кеширование завершено")


if __name__ == "__main__":
    main()