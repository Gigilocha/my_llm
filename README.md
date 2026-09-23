# my-llm

Мультиязычная (RU/EN/code) GPT-style языковая модель ~126M параметров, обучаемая с нуля. Проект построен как воспроизводимый experiment pipeline: конфигурация, данные, токенизатор и архитектура модели разделены на независимые, тестируемые компоненты.

## Идея проекта

Это не обёртка над готовыми весами и не fine-tuning чужой модели — каждый компонент (токенизатор, механизм внимания, слои модели) реализован с нуля, с осознанным выбором архитектурных решений на каждом шаге, а не копированием референсов без понимания.

Ключевой принцип разработки — **reconstruction method**: перед написанием любого кода сначала формулируется словами, что он должен делать и почему, а сам код пишется по памяти/логике, без копирования готовых решений построчно.

## Архитектура модели

- **Механизм внимания:** Grouped Query Attention (GQA) — 12 query-голов, 4 key/value-головы
- **Позиционное кодирование:** RoPE (Rotary Positional Embeddings)
- **Нормализация:** RMSNorm (pre-norm) + опциональная QK-norm внутри attention
- **MLP:** SwiGLU
- **Контекст:** 6144 токена
- **Параметры:** ~126M (hidden_size=768, num_layers=12, vocab_size=65536)
- **KV-cache:** поддержка инкрементального инференса

## Токенизатор

Byte-level BPE, обучен с нуля на мультиязычном корпусе (60% RU / 20% EN / 20% code), vocab_size=65536. Реализован через HuggingFace `tokenizers` + `transformers` для полной совместимости с `AutoTokenizer.from_pretrained(...)`.

## Данные

Потоковая (streaming) загрузка данных из HuggingFace Hub без полного скачивания корпусов:

- **Русский:** FineWeb-2, Wikipedia
- **Английский:** FineWeb
- **Код:** Stack-Edu

Источники смешиваются по заданным пропорциям через `interleave_datasets`, с возможностью локального кеширования в формате Parquet для переиспользования между экспериментами.

### SFT

Формат: `messages` (роль + контент). Смешивание — по двум потокам (RU / EN), код присутствует как категория внутри русского датасета. Пропорции контролируются через `interleave_datasets` с явными вероятностями.

### RLFT

Реализован **DPO** (Direct Preference Optimization) как первый шаг. В планах — **RLVR** (Reinforcement Learning with Verifiable Rewards) для кода и математики: генерация ответов моделью → проверка верификатором (unit-тесты, сверка числа) → бинарная награда.

## Бенчмарки

| Направление | Датасет | Что проверяет |
|---|---|---|
| Русский | `yandex/WikiWebFacts` | Фактология (QA) |
| Русский | `t-tech/ruMMLU-pro` | Знания, 57 областей |
| Русский | `MERA-evaluation/ruHumanEval` | Генерация кода (pass@k) |
| Русский | `MERA-evaluation/ruCodeEval` | Генерация кода (приватный) |
| Английский | `TIGER-Lab/MMLU-Pro` | Знания, 57 областей |
| Английский | `openai/gsm8k` | Математические рассуждения |

## Стек

- **Конфигурация:** Pydantic + pydantic-settings, YAML-конфиги, разделённые по назначению (данные / токенизатор / модель / обучение / движок)
- **Логирование:** stdlib `logging` с цветным форматированием и ротацией файлов
- **Трекинг экспериментов:** MLflow
- **Пакетный менеджер:** [uv](https://github.com/astral-sh/uv)
- **Тесты:** pytest
- **Модель:** PyTorch

## Структура проекта

configs/ # YAML-конфигурации (данные, токенизатор, модель, обучение)
core/
common/ # config.py, logger.py — общая инфраструктура
model/ # attention.py, mlp.py, block.py, model.py
dataset.py # загрузка и смешивание данных
tokenizer.py # обучение и использование токенизатора
scripts/ # entrypoint-скрипты (train_tokenizer.py, train_sft.py, train_rlft.py, ...)
tests/ # pytest-тесты
outputs/ # обученные артефакты (токенизатор, чекпоинты, логи, mlflow)

## Запуск

```bash
uv sync
uv run python scripts/train_tokenizer.py
uv run python scripts/eval_tokenizer.py
uv run python scripts/train_sft.py
uv run python scripts/train_rlft.py
uv run pytest tests/ -v
```

## MLflow UI

# Windows (PowerShell)
```
$env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri file:///E:/AI/Projects/my_llm/outputs/mlflow/mlruns
```

## Статус

☑ Инфраструктура конфигурации (Pydantic + YAML)

☑ Логирование

☑ Пайплайн загрузки и смешивания данных

☑ Обучение и оценка токенизатора (BPE, vocab=65536)

☑ Attention (GQA + RoPE + QK-norm)

☑ SwiGLU MLP

☑ TransformerBlock

☑ Полная модель (embedding + N блоков + LM head)

☑ KV-cache

☑ Pretrain dataloader и цикл обучения

☑ SFT (train + eval)

☑ RLFT (DPO)

□ RLVR (код, математика)

□ Бенчмарки (интеграция)

## Автор
Разработка ведётся в рамках самостоятельного изучения ML/LLM engineering.