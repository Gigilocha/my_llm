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

## Токенизатор

Byte-level BPE, обучен с нуля на мультиязычном корпусе (60% RU / 20% EN / 20% code), vocab_size=65536. Реализован через HuggingFace `tokenizers` + `transformers` для полной совместимости с `AutoTokenizer.from_pretrained(...)`.

## Данные

Потоковая (streaming) загрузка данных из HuggingFace Hub без полного скачивания корпусов:
- Русский: FineWeb-2, Wikipedia
- Английский: FineWeb
- Код: Stack-Edu

Источники смешиваются по заданным пропорциям через `interleave_datasets`, с возможностью локального кеширования в формате Parquet для переиспользования между экспериментами.

## Стек

- **Конфигурация:** Pydantic + pydantic-settings, YAML-конфиги, разделённые по назначению (данные / токенизатор / модель / обучение / движок)
- **Логирование:** stdlib `logging` с цветным форматированием и ротацией файлов
- **Пакетный менеджер:** [uv](https://github.com/astral-sh/uv)
- **Тесты:** pytest
- **Модель:** PyTorch

## Структура проекта

```
configs/          # YAML-конфигурации (данные, токенизатор, модель, обучение)
core/
  common/          # config.py, logger.py — общая инфраструктура
  model/           # attention.py, mlp.py, block.py, model.py
  dataset.py       # загрузка и смешивание данных
  tokenizer.py     # обучение и использование токенизатора
scripts/           # entrypoint-скрипты (train_tokenizer.py, eval_tokenizer.py, ...)
tests/             # pytest-тесты
outputs/           # обученные артефакты (токенизатор, чекпоинты, логи)
```

## Запуск

```bash
uv sync
uv run python scripts/train_tokenizer.py
uv run python scripts/eval_tokenizer.py
uv run pytest tests/ -v
```

## Статус

- [x] Инфраструктура конфигурации (Pydantic + YAML)
- [x] Логирование
- [x] Пайплайн загрузки и смешивания данных
- [x] Обучение и оценка токенизатора (BPE, vocab=65536)
- [x] Attention (GQA + RoPE + QK-norm)
- [x] SwiGLU MLP
- [ ] TransformerBlock
- [ ] Полная модель (embedding + N блоков + LM head)
- [ ] Pretrain dataloader и цикл обучения
- [ ] SFT / RLFT

## Автор

Разработка ведётся в рамках самостоятельного изучения ML/LLM engineering.
