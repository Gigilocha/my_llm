# tests/test_logger.py
import logging
import pytest
from src.common.logger import setup_logger


def _set_logger_env(monkeypatch, tmp_path, log_to_console="true", log_to_file="false"):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("DTYPE", "float32")
    monkeypatch.setenv("CONFIGS_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("LOG_TO_CONSOLE", log_to_console)
    monkeypatch.setenv("LOG_TO_FILE", log_to_file)
    monkeypatch.setenv("LOG_FILE_PATH", "outputs/logs/test.log")
    monkeypatch.setenv("LOG_MAX_BYTES", "1000000")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "3")


# --- 1) Количество handlers соответствует флагам ---
def test_setup_logger_console_only(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path, log_to_console="true", log_to_file="false")
    logger = setup_logger("test_console_only")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logger_console_and_file(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path, log_to_console="true", log_to_file="true")
    logger = setup_logger("test_console_and_file")
    assert len(logger.handlers) == 2


# --- 2) Уровень логирования применился ---
def test_setup_logger_level_applied(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path)
    logger = setup_logger("test_level")
    assert logger.level == logging.DEBUG


# --- 3) Файл лога реально создаётся на диске ---
def test_setup_logger_creates_file(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path, log_to_console="false", log_to_file="true")
    logger = setup_logger("test_file_creation")
    logger.info("test message")

    from src.common.config import PROJECT_ROOT
    log_path = PROJECT_ROOT / "outputs/logs/test.log"
    assert log_path.exists()

    # закрываем handlers перед удалением — иначе Windows не даст удалить занятый файл
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    log_path.unlink()


# --- 4) propagate = False установлен ---
def test_setup_logger_does_not_propagate(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path)
    logger = setup_logger("test_propagate")
    assert logger.propagate is False