# tests/test_config.py
import pytest
from pydantic import ValidationError
from src.common.config import get_env_settings, get_config


def _set_base_env(monkeypatch, tmp_path):
    """Базовые обязательные переменные, нужны почти всем тестам."""
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("DTYPE", "float32")
    monkeypatch.setenv("CONFIGS_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))


# --- 1) Правильное определение полей ---
def test_env_settings_field_mapping(monkeypatch, tmp_path):
    _set_base_env(monkeypatch, tmp_path)
    env = get_env_settings()
    assert env.device == "cpu"
    assert env.dtype == "float32"
    assert env.configs_dir == tmp_path


# --- 2) Обязательные поля реально обязательны ---
def test_env_settings_missing_required_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("DEVICE", raising=False) # намеренно не ставим значение DEVICE
    monkeypatch.setenv("DTYPE", "float32")
    monkeypatch.setenv("CONFIGS_DIR", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path))
    with pytest.raises(ValidationError):
        get_env_settings()


DATA_YAML = """
pre_training_data:
  max_shard: 100
  seed: 42
  val_split_ratio: 0.01
  rus_quantity: 60
  en_quantity: 20
  code_quantity: 20
  rus_sources:
    - dataset_name: test/rus
      split: train
      weight: 1.0
  en_sources: []
  code_sources: []
sft_data:
  rus_sources: []
  en_sources: []
  code_sources: []
rlft_data:
  rus_sources: []
  en_sources: []
  code_sources: []
"""

TOKENIZER_YAML = """
algorithm: bpe
library: tokenizers
vocab_size: 65536
train_sample_size: 1000
special_tokens:
  bos: "<|bos|>"
  eos: "<|eos|>"
split_pattern: "test_pattern"
"""


# --- 3) Правильная загрузка yaml: путь найден, файл распарсен ---
def test_get_config_loads_yaml_files(monkeypatch, tmp_path):
    (tmp_path / "data_config.yaml").write_text(DATA_YAML, encoding="utf-8")
    (tmp_path / "tokenizer_config.yaml").write_text(TOKENIZER_YAML, encoding="utf-8")
    _set_base_env(monkeypatch, tmp_path)

    cfg = get_config()
    assert cfg.tokenizer.vocab_size == 65536
    assert cfg.data.pre_training_data.rus_quantity == 60


# --- 4) Вложенные поля/списки собираются верно ---
def test_get_config_nested_fields(monkeypatch, tmp_path):
    (tmp_path / "data_config.yaml").write_text(DATA_YAML, encoding="utf-8")
    (tmp_path / "tokenizer_config.yaml").write_text(TOKENIZER_YAML, encoding="utf-8")
    _set_base_env(monkeypatch, tmp_path)

    cfg = get_config()
    assert len(cfg.data.pre_training_data.rus_sources) == 1
    assert cfg.data.pre_training_data.rus_sources[0].dataset_name == "test/rus"
    assert cfg.tokenizer.special_tokens["bos"] == "<|bos|>"


# --- 5) Кеширование: один и тот же объект ---
def test_get_config_is_cached(monkeypatch, tmp_path):
    (tmp_path / "data_config.yaml").write_text(DATA_YAML, encoding="utf-8")
    (tmp_path / "tokenizer_config.yaml").write_text(TOKENIZER_YAML, encoding="utf-8")
    _set_base_env(monkeypatch, tmp_path)

    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2