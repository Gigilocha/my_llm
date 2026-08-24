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

    # Логгер
    log_level: str 
    log_to_console: bool
    log_to_file: bool
    log_file: str
    log_max_bytes: int = 50_000_000
    log_backup_count: int = 5



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
# class ModelConfig(BaseModel):


# Обучение
# Класс конфигурации обучения
# class TrainingConfig(BaseModel):


# Общий конфиг для всего
class ExperimentConfig(BaseModel):
    env: EnvSettings
    data: DataConfig
    tokenizer: TokenizerConfig


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
    )


