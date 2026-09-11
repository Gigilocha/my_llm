import pytest


MODEL_YAML = """
model:
  hidden_size: 32
  num_layers: 2
  vocab_size: 300
  max_position_embeddings: 64
  norm_eps: 0.000001
attention:
  num_heads: 4
  num_kv_heads: 2
  head_dim: 8
  rope_theta: 10000.0
  use_qk_norm: true
  qk_norm_eps: 0.000001
mlp:
  intermediate_size: 64
"""

TRAINING_YAML = """
pre_training:
  max_len: 16
  batch_size: 2
  gradient_accumulation_steps: 1
  max_steps: 10
  learning_rate: 0.001
  warmup_steps: 1
  min_learning_rate: 0.0001
  grad_clip_norm: 1.0
  weight_decay: 0.01
  eval_interval: 5
  checkpoint_interval: 5
sft:
  max_len: 16
  batch_size: 2
  gradient_accumulation_steps: 1
  max_steps: 10
  learning_rate: 0.00005
  warmup_steps: 1
  min_learning_rate: 0.000005
  grad_clip_norm: 1.0
  weight_decay: 0.01
  eval_interval: 5
  checkpoint_interval: 5
rlft: {}
"""


def _monitoring_yaml(log_level="info", log_to_console=True, log_to_file=False,
                      log_file_path="outputs/logs/test.log", log_max_bytes=1_000_000,
                      log_backup_count=3) -> str:
    return f"""
memory_monitoring:
  enabled: true
  interval_steps: 50
  log_vram: true
  log_ram: true
  log_allocated: true
  log_reserved: true
speed_monitoring:
  enabled: true
  interval_steps: 50
  log_tokens_per_sec: true
  log_steps_per_sec: true
gradient_monitoring:
  enabled: true
  interval_steps: 50
  log_grad_norm: true
  log_grad_histogram: false
logging:
  log_level: "{log_level}"
  log_to_console: {str(log_to_console).lower()}
  log_to_file: {str(log_to_file).lower()}
  log_file_path: "{log_file_path}"
  log_max_bytes: {log_max_bytes}
  log_backup_count: {log_backup_count}
mlflow:
  enabled: true
  tracking_uri: "outputs/mlflow/mlruns"
  experiment_name: "test"
  log_artifacts: false
  log_model: false
  log_params: false
  log_metrics: false
  log_tags:
    model_type: "test"
"""


# Пишет все YAML-конфиги, которые требует get_config(), кроме data/tokenizer —
# их пишет вызывающий тест сам, т.к. их содержимое обычно и есть предмет проверки
def write_required_configs(tmp_path, **logging_overrides) -> None:
    (tmp_path / "model_config.yaml").write_text(MODEL_YAML, encoding="utf-8")
    (tmp_path / "training_config.yaml").write_text(TRAINING_YAML, encoding="utf-8")
    (tmp_path / "monitoring_config.yaml").write_text(_monitoring_yaml(**logging_overrides), encoding="utf-8")
    (tmp_path / "engine_config.yaml").write_text("", encoding="utf-8")  # все поля EngineConfig опциональны


# Очистка кеша
@pytest.fixture(autouse=True)
def _clean_config_cache():
    from src.common.config import get_env_settings, get_config
    get_env_settings.cache_clear()
    get_config.cache_clear()
    yield


# Ограничение на использование реального енв файла
@pytest.fixture(autouse=True)
def _isolate_from_real_env_file(monkeypatch, tmp_path):
    # почему: реальный .env в корне проекта не должен утекать в тесты —
    # без этого EnvSettings может найти значения из настоящего .env,
    # даже если тест явно их не задавал через monkeypatch.setenv
    monkeypatch.chdir(tmp_path)
    yield


import logging

# Закрытие handlers после каждого теста
@pytest.fixture(autouse=True)
def _close_all_loggers():
    yield
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)