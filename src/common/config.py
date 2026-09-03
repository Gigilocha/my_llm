from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Определение корня проекта
PROJECT_ROOT = Path(__file__).parents[2].resolve()


# Настройки из .env
class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore'
    )

    # Устройство
    device: str 
    dtype: str 

    # Директории
    configs_dir: Path 
    data_dir: Path
    outputs_dir: Path


# Мониторинг
# Конфиг мониторинга памяти
class MemoryMonitoring(BaseModel):
    enabled: bool
    interval_steps: int          # как часто логировать (шаги)
    log_vram: bool
    log_ram: bool
    log_allocated: bool
    log_reserved: bool

# Конфиг мониторинга скорости генерации
class SpeedMonitoring(BaseModel):
    enabled: bool
    interval_steps: int
    log_tokens_per_sec: bool
    log_steps_per_sec: bool

# Конфиг логирования градиентов
class GradientMonitoring(BaseModel):
    enabled: bool
    interval_steps: int
    log_grad_norm: bool
    log_grad_histogram: bool

# Конфиг логирования logging
class Logging(BaseModel):
    log_level: str
    log_to_console: bool
    log_to_file: bool
    log_file_path: str
    log_max_bytes: int
    log_backup_count: int

# Конфиг логирования mlflow
class MlFlow(BaseModel):
    enabled: bool
    tracking_uri: str
    experiment_name: str
    log_artifacts: bool
    log_model: bool
    log_params: bool
    log_metrics: bool
    log_tags: dict[str]

# Общий класс для мониторинга
class Monitoring(BaseModel):
    meory_monitoring_config: MemoryMonitoring 
    speed_monitoring_confgig: SpeedMonitoring
    gradient_monitoring_config: GradientMonitoring
    logging_config: Logging
    mlflow_config: MlFlow


# Данные
# Конфиг ресурсов данных
class DataSource(BaseModel):
    dataset_name: str
    subset: str | None = None
    split: str = "train"
    weight: float = 1.0

# Конфиг разделения данных
class SplitData(BaseModel):
    rus_sources: list[DataSource] = []
    en_sources: list[DataSource] = []
    code_sources: list[DataSource] = []

# Конфиг данных для первичного обучения
class PretrainData(SplitData):
    max_shard: int
    seed: int
    val_split_ratio: float
    rus_quantity: int
    en_quantity: int
    code_quantity: int
    rus_cache_docs: int      
    en_cache_docs: int       
    code_cache_docs: int

# Конфиг данных конечный
class DataConfig(BaseModel):
    pre_training_data: PretrainData
    sft_data: SplitData
    rlft_data: SplitData


# Токенизатор
class TokenizerConfig(BaseModel):
    algorithm: str 
    library: str 
    vocab_size: int
    train_sample_size: int
    special_tokens: dict
    split_pattern: str 


# Модель 
# Класс конфигурации модели
class ModelConfig(BaseModel):
    hidden_size: int
    num_layers: int
    vocab_size: int 
    max_position_embeddings: int   # контекст, ~6144 или 8192 (степень двойки/512 удобнее)
    norm_eps: float           # eps для RMSNorm (pre-norm блоков)
    window_pattern: str = "L"  # заглушка на будущее, все full attention пока

# Класс конфигурации внимания
class AttentionConfig(BaseModel):
    num_heads: int          # query heads
    num_kv_heads: int        # GQA — меньше, чем num_heads (например, 12 и 4)
    head_dim: int
    rope_theta: float        # база RoPE, обычно 10000.0
    use_qk_norm: bool
    qk_norm_eps: float       # eps для QK-norm (0 или отсутствие поля = выключено? или отдельный bool)

# Класс конфигурации MLP
class MLPConfig(BaseModel):
    intermediate_size: int    # ширина SwiGLU-слоя (обычно ~2.67x hidden_size из-за gate+up+down в SwiGLU)

# Конфиг модели конечный
class GPTConfig(BaseModel):
    model: ModelConfig
    attention: AttentionConfig
    mlp: MLPConfig


# Обучение
# Класс конфигурации базового обучения
class PretrainingConfig(BaseModel):
    max_len: int
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    warmup_steps: int
    min_learning_rate: float
    grad_clip_norm: float
    weight_decay: float
    eval_interval: int
    checkpoint_interval: int

# Класс конфигурации тонкой настройки (sft)
class SFTConfig(BaseModel):
    pass

# Класс конфигурации тонкой настройки (rlft)
class RLFTConfig(BaseModel):
    pass

# Общий класс для обучения
class TrainingConfig(BaseModel):  
    pre_training: PretrainingConfig
    sft: SFTConfig
    rlft: RLFTConfig


# Общий конфиг для всего
class ExperimentConfig(BaseModel):
    env: EnvSettings
    monitoring: Monitoring
    data: DataConfig
    tokenizer: TokenizerConfig
    model: GPTConfig
    training: TrainingConfig


# Загрузка .yaml по названию
def _load_yaml(env: EnvSettings, filename: str) -> dict:
    path = PROJECT_ROOT / env.configs_dir / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Получение настроек
@lru_cache
def get_env_settings() -> EnvSettings:
    return EnvSettings()

# Получение конфига
@lru_cache
def get_config() -> ExperimentConfig:
    env = get_env_settings()  # переиспользуем закешированный EnvSettings
    return ExperimentConfig(
        env=env,
        data=DataConfig(**_load_yaml(env, "data_config.yaml")),
        tokenizer=TokenizerConfig(**_load_yaml(env, "tokenizer_config.yaml")),
        model=GPTConfig(**_load_yaml(env, "model_config.yaml")),
        training=TrainingConfig(**_load_yaml(env, "training_config.yaml")),
    )


