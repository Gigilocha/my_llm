from pathlib import Path
import re
import logging
from logging.handlers import RotatingFileHandler
from src.common.config import get_config, PROJECT_ROOT  # ← меняем импорт


"""
Заметка:
    Формат логов взят из оригинального проекта Андрея Карпатого NanoChat c выделением чисел и процентов.
"""


# Формат логгера
# Класс формата для логгера
class ColoredFormatter(logging.Formatter):
    # ANSI коды цветов
    COLORS = {
        'DEBUG': '\033[36m',    # Синий
        'INFO': '\033[32m',     # Зеленый
        'WARNING': '\033[33m',  # Желтый
        'ERROR': '\033[31m',    # Красный
        'CRITICAL': '\033[35m', # Малиновый
    }
    BOLD = '\033[1m'
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord):
        levelname = record.levelname
        
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        message = super().format(record)

        if levelname == "INFO":
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        
        return message


# Сетап логгера
def setup_logger(name: str) -> logging.Logger:
    # Берем конфиг из Monitoring
    config = get_config()
    logging_cfg = config.monitoring.logging  # ← новые настройки
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, logging_cfg.log_level.upper()))
    logger.propagate = False

    # Логи в консоль
    if logging_cfg.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    # Логи в файл
    if logging_cfg.log_to_file:
        log_path = PROJECT_ROOT / logging_cfg.log_file_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=logging_cfg.log_max_bytes,
            backupCount=logging_cfg.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger