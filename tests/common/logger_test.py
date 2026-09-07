# tests/test_logger.py
import logging
import pytest
from src.common.logger import setup_logger
from tests.conftest import write_required_configs, MODEL_YAML, TRAINING_YAML

# data/tokenizer конфиги минимальные — get_config() требует их присутствия,
# но логгеру их содержимое не важно
DATA_YAML = "pre_training_data:\n  max_shard: 1\n  seed: 1\n  val_split_ratio: 0.1\n  rus_quantity: 1\n  en_quantity: 1\n  code_quantity: 1\n  rus_cache_docs: 1\n  en_cache_docs: 1\n  code_cache_docs: 1\nsft_data: {}\nrlft_data: {}\n"
TOKENIZER_YAML = 'algorithm: bpe\nlibrary: tokenizers\nvocab_size: 300\ntrain_sample_size: 10\nspecial_tokens:\n  bos: "<|bos|>"\n  eos: "<|eos|>"\nsplit_pattern: "test"\n'


def _set_logger_env(monkeypatch, tmp_path, log_to_console=True, log_to_file=False):
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("DTYPE", "float32")
    monkeypatch.setenv("CONFIGS_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))

    (tmp_path / "data_config.yaml").write_text(DATA_YAML, encoding="utf-8")
    (tmp_path / "tokenizer_config.yaml").write_text(TOKENIZER_YAML, encoding="utf-8")
    write_required_configs(
        tmp_path,
        log_level="debug",
        log_to_console=log_to_console,
        log_to_file=log_to_file,
        log_file_path="outputs/logs/test.log",
    )


# --- 1) Количество handlers соответствует флагам ---
def test_setup_logger_console_only(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path, log_to_console=True, log_to_file=False)
    logger = setup_logger("test_console_only")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logger_console_and_file(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path, log_to_console=True, log_to_file=True)
    logger = setup_logger("test_console_and_file")
    assert len(logger.handlers) == 2


# --- 2) Уровень логирования применился ---
def test_setup_logger_level_applied(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path)
    logger = setup_logger("test_level")
    assert logger.level == logging.DEBUG


# --- 3) Файл лога реально создаётся на диске ---
def test_setup_logger_creates_file(monkeypatch, tmp_path):
    _set_logger_env(monkeypatch, tmp_path, log_to_console=False, log_to_file=True)
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
