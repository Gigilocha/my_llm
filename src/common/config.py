from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field, model_validator
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

    # Устройство. "auto" (по умолчанию) — определяется в device.py:resolve_device().
    # Можно задать явно (cuda/cpu) в .env — тогда используется как есть
    device: str = "auto"
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
    log_tags: dict[str, str]

# Общий класс для мониторинга
class Monitoring(BaseModel):
    memory_monitoring: MemoryMonitoring
    speed_monitoring: SpeedMonitoring
    gradient_monitoring: GradientMonitoring
    logging: Logging
    mlflow: MlFlow


# Данные
# Конфиг ресурсов данных
class DataSource(BaseModel):
    dataset_name: str
    subset: str | None = None
    split: str = "train"
    weight: float = 1.0
    # Для датасетов с вложенной repo-структурой (одна строка = репозиторий,
    # файлы лежат в списке files[]), например HuggingFaceCode/stack-v3-train:
    # значение поля "language" внутри files[], по которому фильтруем конкретный
    # язык программирования. None — источник плоский, фильтрация не нужна
    language_filter: str | None = None

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

# Один источник SFT-данных. instruction/output — обязательные поля (после
# маппинга), input/system — опциональные. Разные датасеты называют колонки
# по-разному (instruction/output, question/answer и т.д.) — вместо того чтобы
# угадывать и жёстко зашивать имена в код, задаём маппинг в конфиге. Если
# угадаем неверно — cache_sft_data() упадёт с понятной ошибкой на первом же
# документе (список реально доступных полей в сообщении), а не молча
# закеширует пустые примеры, как было с stack-v3-train до фикса
class SFTSource(BaseModel):
    dataset_name: str
    subset: str | None = None
    split: str = "train"
    weight: float = 1.0

    # Формат А (предпочтительный): готовый список сообщений — [{"role": ...,
    # "content": ...}, ...] (ShareGPT/OpenAI-style chat format). Так хранит
    # большинство современных SFT-датасетов (T-Wix-instag, EagleSFT — оба
    # в этом формате). Поддерживает multi-turn "из коробки"
    messages_field: str | None = None

    # Формат Б: плоские instruction/output — для источников без messages
    instruction_field: str | None = None
    input_field: str | None = None
    output_field: str | None = None
    system_field: str | None = None

    # Фильтр по значению плоского поля-метки (например supertag=code у
    # T-Wix-instag) — позволяет завести ОДИН датасет как НЕСКОЛЬКО источников
    # с разными весами (код отдельно от остального), не задваивая примеры.
    # filter_values — список: документ проходит, если значение поля есть в списке
    filter_field: str | None = None
    filter_values: list[str] | None = None

    # Явное имя для папки кеша — обязательно, когда один dataset_name встречается
    # в конфиге больше одного раза (иначе разные filter_values затрут друг друга
    # на диске: путь строится из dataset_name, а filter_values в нём не участвует)
    cache_name: str | None = None

    @model_validator(mode="after")
    def _check_format_configured(self) -> "SFTSource":
        has_messages = self.messages_field is not None
        has_flat = self.instruction_field is not None and self.output_field is not None
        if has_messages == has_flat:  # ни один не задан, или заданы оба разом
            raise ValueError(
                f"{self.dataset_name}: укажи либо messages_field, либо оба "
                f"instruction_field/output_field — ровно один формат, не оба и не ни одного"
            )
        if (self.filter_field is None) != (self.filter_values is None):
            raise ValueError(
                f"{self.dataset_name}: filter_field и filter_values задаются вместе или не задаются вообще"
            )
        return self

# Источник preference-данных для DPO: на каждый пример нужны промпт и ДВА
# ответа — предпочтительный (chosen) и отвергнутый (rejected). Никакой
# reward-модели не требуется, предпочтение задано прямо в данных
class DPOSource(BaseModel):
    dataset_name: str
    subset: str | None = None
    split: str = "train"
    weight: float = 1.0

    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"

    # chosen/rejected бывают либо строкой, либо списком messages (тогда берём
    # content последнего хода assistant). prompt при as_messages=True тоже
    # может быть списком messages — берём его как историю диалога целиком
    as_messages: bool = False

    filter_field: str | None = None
    filter_values: list[str] | None = None
    cache_name: str | None = None

    @model_validator(mode="after")
    def _check_filter_configured(self) -> "DPOSource":
        if (self.filter_field is None) != (self.filter_values is None):
            raise ValueError(
                f"{self.dataset_name}: filter_field и filter_values задаются вместе или не задаются вообще"
            )
        return self


class DPOData(BaseModel):
    sources: list[DPOSource] = []
    cache_docs_per_source: int = 50_000
    val_split_ratio: float = 0.01
    seed: int = 42


class SFTData(BaseModel):
    sources: list[SFTSource] = []
    cache_docs_per_source: int = 100_000
    val_split_ratio: float = 0.01
    seed: int = 42

# Конфиг данных конечный
class DataConfig(BaseModel):
    pre_training_data: PretrainData
    sft_data: SFTData
    rlft_data: DPOData


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
    # Muon (для 2D-весов трансформера) — с дефолтами, чтобы не требовать правки
    # существующего training_config.yaml. muon_lr обычно на порядок больше AdamW lr
    muon_learning_rate: float = 0.02
    muon_weight_decay: float = 0.0
    muon_momentum: float = 0.95

# Класс конфигурации тонкой настройки (sft)
class SFTConfig(BaseModel):
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
    muon_learning_rate: float = 0.02
    muon_weight_decay: float = 0.0
    muon_momentum: float = 0.95

# Класс конфигурации тонкой настройки (rlft)
class RLFTConfig(BaseModel):
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
    # beta — насколько сильно политика может отходить от reference-модели.
    # Меньше beta = свободнее отход (и выше риск деградации языка), больше =
    # ближе к SFT-модели. 0.1 — типичное значение из статьи DPO
    beta: float = 0.1
    muon_learning_rate: float = 0.003
    muon_weight_decay: float = 0.0
    muon_momentum: float = 0.95

# Общий класс для обучения
class TrainingConfig(BaseModel):  
    pre_training: PretrainingConfig
    sft: SFTConfig
    rlft: RLFTConfig


# Общий конфиг для всего
# Конфиг движка генерации — параметры сэмплирования по умолчанию и выбор
# движка (наивный без KV-кеша vs GenerationEngine с кешем). Скрипты могут
# переопределять отдельные параметры через CLI (см. base_eval.py) — эти
# значения используются как дефолт, когда параметр явно не передан
class EngineConfig(BaseModel):
    use_kv_cache: bool = True
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float | None = None
    repetition_penalty: float = 1.3


class ExperimentConfig(BaseModel):
    env: EnvSettings
    monitoring: Monitoring
    data: DataConfig
    tokenizer: TokenizerConfig
    model: GPTConfig
    training: TrainingConfig
    engine: EngineConfig


# Загрузка .yaml по названию
def _load_yaml(env: EnvSettings, filename: str) -> dict:
    path = PROJECT_ROOT / env.configs_dir / filename
    with open(path, encoding="utf-8") as f:
        # yaml.safe_load на пустом файле возвращает None, а не {} —
        # без этого EngineConfig(**None) упал бы с TypeError
        return yaml.safe_load(f) or {}


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
        monitoring=Monitoring(**_load_yaml(env, "monitoring_config.yaml")),
        data=DataConfig(**_load_yaml(env, "data_config.yaml")),
        tokenizer=TokenizerConfig(**_load_yaml(env, "tokenizer_config.yaml")),
        model=GPTConfig(**_load_yaml(env, "model_config.yaml")),
        training=TrainingConfig(**_load_yaml(env, "training_config.yaml")),
        engine=EngineConfig(**_load_yaml(env, "engine_config.yaml")),
    )


