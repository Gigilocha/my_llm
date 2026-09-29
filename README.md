# my-llm

Воспроизводимый experiment pipeline для полного цикла работы с языковыми моделями:  
от обучения токенизатора и архитектуры с нуля → pretrain → SFT → RLFT (DPO) → инференс и оценка.

Каждый компонент (конфигурация, данные, токенизатор, модель, обучение, движок генерации) выделен в независимый, тестируемый модуль.  
Проект не является обёрткой над готовыми весами — всё реализовано самостоятельно.

## Идея

Ключевой принцип — **reconstruction method**:  
перед написанием любого кода сначала формулируется словами, что он должен делать и почему.  
Код пишется по логике, а не копируется построчно из референсов.

Цель пайплайна — дать полный контроль над каждым этапом жизненного цикла LLM и возможность легко менять:
- пропорции данных
- архитектуру модели
- гиперпараметры обучения
- стратегию выравнивания (SFT → DPO → RLVR)
- способ инференса

## Компоненты пайплайна

### 1. Конфигурация
- Pydantic + pydantic-settings
- YAML-конфиги, разделённые по назначению:
  - `data_config.yaml`
  - `tokenizer_config.yaml`
  - `model_config.yaml`
  - `training_config.yaml`
  - `engine_config.yaml`
  - `monitoring_config.yaml`

Все параметры экспериментов задаются декларативно. Можно запускать разные конфигурации без изменения кода.

### 2. Данные
- Потоковая (streaming) загрузка с HuggingFace Hub
- Смешивание источников через `interleave_datasets` с явными вероятностями
- Локальный кеш в Parquet для переиспользования между экспериментами
- Поддержка pretrain-микса, SFT (формат `messages`) и preference-данных для DPO

### 3. Токенизатор
- Byte-level BPE, обучаемый с нуля
- Полная совместимость с HuggingFace `AutoTokenizer.from_pretrained(...)`
- Отдельные скрипты обучения и оценки токенизатора

### 4. Модель
Модульная GPT-style архитектура:
- Grouped Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- RMSNorm (pre-norm) + опциональная QK-norm
- SwiGLU MLP
- Weight tying
- Поддержка causal + padding масок
- KV-cache для инкрементального инференса

Архитектура полностью конфигурируема через YAML (размер, количество слоёв, голов, контекст и т.д.).

### 5. Обучение
Полный цикл:
1. **Pretrain** — обучение с нуля на смешанном корпусе
2. **SFT** — supervised fine-tuning на instruction-данных
3. **RLFT (DPO)** — Direct Preference Optimization
4. *(в планах)* **RLVR** — Reinforcement Learning with Verifiable Rewards (код + математика)

Каждый этап имеет свой train/eval цикл, checkpointing и логирование.

### 6. Движок инференса
- Naive generation
- Generation с KV-cache
- Поддержка temperature / top-k / top-p / repetition penalty
- Интерактивный чат (`scripts/chat.py`)
- Адаптер под lm-eval

### 7. Трекинг и мониторинг
- MLflow (параметры, метрики, артефакты, модели)
- Цветное логирование + ротация файлов
- Замер скорости, памяти, градиентов

## Структура проекта

```
configs/                  # YAML-конфигурации
src/
  common/                 # конфиг, логгер, device, mlflow-утилиты
  data/                   # загрузка, смешивание, форматы (pretrain / SFT / DPO)
  tokenizer/              # обучение и использование токенизатора
  model/                  # Attention, MLP, Block, Transformer
  training/               # train/eval steps, optimizer, lr schedule, checkpoint, DPO
  engine/                 # generation, KV-cache, lm-eval adapter
scripts/                  # entrypoint-скрипты всех этапов
outputs/                  # чекпоинты, токенизатор, логи, mlflow
```

## Быстрый старт

```bash
uv sync

# 1. Обучение токенизатора
uv run python scripts/tokenizer_train.py
uv run python scripts/tokenizer_eval.py

# 2. Pretrain
uv run python scripts/base_data_build.py
uv run python scripts/base_train.py
uv run python scripts/base_eval.py

# 3. SFT
uv run python scripts/sft_data_build.py
uv run python scripts/sft_train.py
uv run python scripts/sft_eval.py

# 4. RLFT (DPO)
uv run python scripts/rlft_data_build.py
uv run python scripts/rlft_train.py

# Интерактивный чат
uv run python scripts/chat.py

# Тесты
uv run pytest tests/ -v
```

## MLflow UI

```bash
# Linux / macOS
mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri file://$(pwd)/outputs/mlflow/mlruns

# Windows (PowerShell)
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri file:///E:/AI/Projects/my_llm/outputs/mlflow/mlruns
```

## Статус пайплайна

| Этап                          | Статус |
|-------------------------------|--------|
| Инфраструктура конфигурации   | ☑     |
| Логирование + MLflow          | ☑     |
| Пайплайн данных (streaming + mix) | ☑  |
| Токенизатор (train + eval)    | ☑     |
| Архитектура модели            | ☑     |
| KV-cache + generation engine  | ☑     |
| Pretrain                      | ☑     |
| SFT (train + eval)            | ☑     |
| RLFT / DPO                    | ☑     |
| RLVR (verifiable rewards)     | □     |
| Интеграция бенчмарков         | □     |
| Оркестратор запуска           | □     |

## Принципы проектирования

- **Модульность** — любой компонент можно заменить, не ломая остальное
- **Конфигурируемость** — все важные решения вынесены в YAML
- **Воспроизводимость** — один и тот же конфиг + данные → одинаковый эксперимент
- **Понятность** — код пишется так, чтобы его можно было разобрать и объяснить
- **Полный цикл** — от сырых данных до работающего чата

## Автор

Разработка ведётся в рамках самостоятельного изучения ML / LLM engineering.

## Связь с автором

Emal: Gigilocha@yandex.ru

Telegram: [t.me/Gigilocha]

Канал: [t.me/SkinAI_KSE]

В канале пишу о ходе работы: к чему пришёл, какие были сложности, какие ошибки допустил и как их решал. Если интересно видеть процесс мышления и принятия решений — заходите.
