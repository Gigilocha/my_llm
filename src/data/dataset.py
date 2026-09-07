from pathlib import Path
from typing import Iterator
from itertools import islice
from collections import deque

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from datasets import interleave_datasets, IterableDataset

from src.common.config import DataSource, PretrainData


# Разворачивает repo-документ (например, HuggingFaceCode/stack-v3-train: одна
# строка = репозиторий, файлы лежат в списке files[]) в отдельные документы —
# по одному на файл нужного языка. Так длина документа = длина файла, а не
# репозитория целиком (что для расчёта средней длины и семплирования важно —
# см. переоценку пропорций микса), и текст сразу лежит в поле "text" единообразно
# с плоскими источниками, дальше по пайплайну ничего не меняется
def _unroll_repo_files(repo_stream, language_filter: str) -> Iterator[dict]:
    for repo in repo_stream:
        for f in repo.get("files", []):
            if f.get("language") == language_filter:
                content = f.get("content") or ""
                if content:
                    yield {"text": content}


# Запуск одного стрима данных из сети.
# source.language_filter задан -> источник имеет вложенную repo-структуру
# (files[] со своим полем language внутри каждого репозитория) и его нужно
# развернуть в плоский поток отдельных файлов нужного языка
def load_source_stream(source: DataSource) -> IterableDataset:
    ds = load_dataset(path=source.dataset_name, name=source.subset, split=source.split, streaming=True)
    if source.language_filter is not None:
        return _unroll_repo_files(ds, source.language_filter)
    return ds


# Функция построения безопасного имени папки датасета
def _safe_dirname(source: DataSource) -> str:
    name = source.dataset_name.replace("/", "_")
    if source.subset:
        name = f"{name}__{source.subset.replace('/', '_')}"
    return name


# Путь к локальной папке с шардами конкретного источника
# stage разделяет данные по этапу обучения (pretrain / sft / rlft),
# т.к. один и тот же источник может кешироваться по-разному для разных этапов
def _local_dir_for_source(source: DataSource, stage: str, language: str, data_dir: Path, split: str) -> Path:
    return data_dir / stage / split / language / _safe_dirname(source)


# Смешивание списка стримов по весам в один — общая логика для сетевых и локальных миксов.
# stopping_strategy="all_exhausted": по умолчанию HF останавливает общий стрим, как только
# ПЕРВЫЙ из источников иссяк (first_exhausted) — при неравных весах и объёмах данных это
# обрубает весь микс ещё до того, как большинство источников дочитаны до конца (проверено:
# 210 честных документов из трёх языков давали всего 145 на выходе). all_exhausted вместо
# этого досэмплирует (с повтором) уже исчерпанные источники, пока не иссякнет последний —
# так реально используются все закешированные данные, а не только их часть
def _weighted_interleave(streams: list[IterableDataset], weights: list[float], seed: int) -> IterableDataset:
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    return interleave_datasets(streams, probabilities=probabilities, seed=seed, stopping_strategy="all_exhausted")


# Кеширование одного источника на диск сразу шардами (Parquet) — один шард = одна часть в RAM,
# а не весь источник целиком (раньше один большой файл на источник приводил к OOM при чтении)
def cache_source_to_disk(
    source: DataSource,
    stage: str,
    language: str,
    data_dir: Path,
    train_docs: int,
    val_docs: int,
    rows_per_shard: int = 50_000,
) -> tuple[Path, Path]:
    train_dir = _local_dir_for_source(source, stage, language, data_dir, "train")
    val_dir = _local_dir_for_source(source, stage, language, data_dir, "val")
    train_marker = train_dir / "_SUCCESS"
    val_marker = val_dir / "_SUCCESS"

    if train_marker.exists() and val_marker.exists():
        return train_dir, val_dir

    # ВАЖНО: train и val пишутся из ОДНОГО непрерывного итератора по стриму, а не
    # из двух независимых. Если открыть стрим дважды (отдельно для train, отдельно
    # для val), оба чтения начинаются с начала источника — val тогда оказывается
    # подмножеством train (полная утечка данных, val перестаёт быть val).
    #
    # ПОРЯДОК: val ПЕРВЫМ, потом train — а не наоборот. Причина: если реальный
    # объём источника (сколько документов физически есть в потоке) меньше train_docs
    # (например, wikipedia на языке меньше, чем рассчитанный rus_cache_docs),
    # запись train исчерпывает stream_iter полностью, и val получает 0 документов
    # молча, без ошибки. val маленький (обычно ~1%) — почти всегда влезает первым,
    # так он не голодает из-за нехватки данных у источника
    stream_iter = iter(load_source_stream(source))

    if val_marker.exists():
        # val с прошлого (возможно прерванного) запуска уже готов — пропускаем
        # val_docs документов, чтобы поставить итератор на позицию train
        _consume(stream_iter, val_docs)
    else:
        val_dir.mkdir(parents=True, exist_ok=True)
        for stale_shard in val_dir.glob("part_*.parquet"):
            stale_shard.unlink()
        written = _write_shards(stream_iter, val_dir, val_docs, rows_per_shard)
        val_marker.write_text(f"docs={written}\n")
        if written < val_docs:
            print(f"⚠️  {source.dataset_name}: источник закончился раньше — val получил {written}/{val_docs} документов")

    if not train_marker.exists():
        train_dir.mkdir(parents=True, exist_ok=True)
        for stale_shard in train_dir.glob("part_*.parquet"):
            stale_shard.unlink()
        written = _write_shards(stream_iter, train_dir, train_docs, rows_per_shard)
        train_marker.write_text(f"docs={written}\n")
        if written < train_docs:
            print(f"⚠️  {source.dataset_name}: источник закончился раньше — train получил {written}/{train_docs} документов "
                  f"(это может исказить реальную долю токенов — проверь data_eval.py после кеширования)")

    return train_dir, val_dir


# Пропустить n элементов итератора, ничего не сохраняя (используется, когда train
# уже закеширован с прошлого запуска, а val — ещё нет: нужно поставить итератор
# на позицию сразу после train, не читая его заново в память)
def _consume(iterator, n: int) -> None:
    deque(islice(iterator, n), maxlen=0)


# Записать до max_docs документов из итератора шардами по rows_per_shard.
# Возвращает фактическое количество записанных документов (может быть меньше
# max_docs, если в источнике закончились данные)
def _write_shards(stream_iter, save_dir: Path, max_docs: int, rows_per_shard: int) -> int:
    shard_index = 0
    total_written = 0

    while total_written < max_docs:
        take = min(rows_per_shard, max_docs - total_written)
        raw_batch = list(islice(stream_iter, take))
        if not raw_batch:
            break

        # Оставляем только текст. Исходные датасеты (fineweb-2/wikipedia/stack-v3 и т.п.)
        # тащат в каждом документе служебные поля с вложенными структурами (struct/list),
        # схема которых плавает от партии к партии (где-то поле None, где-то заполнено) —
        # при чтении нескольких таких шардов разом PyArrow не может привести вложенные
        # типы между чанками и падает с ArrowNotImplementedError. Нам эти поля не нужны
        batch = [{"text": (doc.get("text") or doc.get("content") or "")} for doc in raw_batch]

        table = pa.Table.from_pylist(batch)
        pq.write_table(table, save_dir / f"part_{shard_index:04d}.parquet")

        total_written += len(batch)
        shard_index += 1

    return total_written


# Кеширование всех источников под свой лимит документов на язык
def cache_pretrain_data(cfg: PretrainData, data_dir: Path, stage: str = "pretrain") -> None:

    language_sources = {
        "rus": (cfg.rus_sources, cfg.rus_cache_docs),
        "en": (cfg.en_sources, cfg.en_cache_docs),
        "code": (cfg.code_sources, cfg.code_cache_docs),
    }

    for language, (sources, max_docs) in language_sources.items():
        for source in sources:
            train_docs = int(max_docs * (1 - cfg.val_split_ratio))
            val_docs = int(max_docs * cfg.val_split_ratio)
            cache_source_to_disk(source, stage, language, data_dir, train_docs=train_docs, val_docs=val_docs)


# Чтение уже закешированного источника с диска — забираем все шарды из папки одним стримом
def load_local_stream(shard_dir: Path) -> IterableDataset:
    shard_files = sorted(str(p) for p in shard_dir.glob("part_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"Нет шардов в {shard_dir}")
    return load_dataset("parquet", data_files=shard_files, split="train", streaming=True)


# --- Сетевые версии (используются, например, tokenizer_train.py — не требуют предварительного кеша) ---

# Смешать источники одного языка по весам (из сети)
def build_language_mix(sources: list[DataSource], seed: int) -> IterableDataset:
    streams = [load_source_stream(source) for source in sources]
    weights = [source.weight for source in sources]
    return _weighted_interleave(streams, weights, seed)


# Сборка микса для Pre-training из разных языков (из сети)
def build_pretrain_mix(cfg: PretrainData) -> IterableDataset:
    rus_mix = build_language_mix(cfg.rus_sources, cfg.seed)
    en_mix = build_language_mix(cfg.en_sources, cfg.seed)
    code_mix = build_language_mix(cfg.code_sources, cfg.seed)

    weights = [cfg.rus_quantity, cfg.en_quantity, cfg.code_quantity]
    return _weighted_interleave([rus_mix, en_mix, code_mix], weights, cfg.seed)


# --- Локальные версии (читают уже закешированные на диске данные) ---

# Смешать источники одного языка по весам (с диска)
def build_language_mix_from_disk(
    sources: list[DataSource],
    stage: str,
    language: str,
    data_dir: Path,
    seed: int,
    split: str = "train"
) -> IterableDataset:
    streams = []
    weights = []
    for source in sources:
        shard_dir = _local_dir_for_source(source, stage, language, data_dir, split)
        # Читаем только полностью закешированные источники (см. _SUCCESS в cache_source_to_disk) —
        # недописанная или прерванная на середине папка не должна попасть в обучение
        if not (shard_dir / "_SUCCESS").exists():
            print(f"⚠️ Предупреждение: {shard_dir} не закеширован (нет _SUCCESS), пропускаем")
            continue
        streams.append(load_local_stream(shard_dir))
        weights.append(source.weight)

    if not streams:
        raise FileNotFoundError(f"Нет данных для {stage}/{split}/{language}")

    return _weighted_interleave(streams, weights, seed)


# Сборка микса для Pre-training из разных языков (с диска)
def build_pretrain_mix_from_disk(
    cfg: PretrainData,
    data_dir: Path,
    stage: str = "pretrain",
    split: str = "train"
) -> IterableDataset:
    rus_mix = build_language_mix_from_disk(cfg.rus_sources, stage, "rus", data_dir, cfg.seed, split)
    en_mix = build_language_mix_from_disk(cfg.en_sources, stage, "en", data_dir, cfg.seed, split)
    code_mix = build_language_mix_from_disk(cfg.code_sources, stage, "code", data_dir, cfg.seed, split)

    weights = [cfg.rus_quantity, cfg.en_quantity, cfg.code_quantity]
    return _weighted_interleave([rus_mix, en_mix, code_mix], weights, cfg.seed)


# Конвертация в текст
def extract_texts(dataset: IterableDataset) -> Iterator[str]:
    for doc in dataset:
        text = (doc.get("text") or doc.get("content") or "").strip()
        if text:
            yield text
