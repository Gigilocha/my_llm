from pathlib import Path
import re
import logging
from logging.handlers import RotatingFileHandler
from core.common import get_env_settings, PROJECT_ROOT


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
        # Сохраняем оригинальный уровень
        levelname = record.levelname
        
        # Добавление цвета к уровню логирования
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        # Формат сообщения лога
        message = super().format(record)

        # Добавление цвета к определенным частям сообщения
        if levelname == "INFO":  # Было "info" (нижний регистр)
            # Выделение числа и процента
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        
        return message


# Сетап логгера
def setup_logger(name: str) -> logging.Logger:
    settings = get_env_settings()
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    logger.propagate = False

    # Логи в консоль
    if settings.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    # Логи в файл
    if settings.log_to_file:
        log_path = PROJECT_ROOT / settings.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger


# Инициализация логгера
logger = setup_logger("my_llm")