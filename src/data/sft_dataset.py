from pathlib import Path
from itertools import islice, chain
from collections import deque

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, interleave_datasets, IterableDataset

from src.common.config import SFTSource, SFTData


def _local_dir_for_sft_source(source: SFTSource, data_dir: Path, split: str) -> Path:
    if source.cache_name:
        name = source.cache_name.replace("/", "_")
    else:
        name = source.dataset_name.replace("/", "_")
        if source.subset:
            name = f"{name}__{source.subset.replace('/', '_')}"
    return data_dir / "sft" / split / name


# Проверяем, что настроенные поля реально существуют в первом документе
# источника — ДО того, как молча закешируем тысячи пустых примеров. Та же
# ошибка, что уже ловили на stack-v3-train (там doc.get("text") возвращал ""
# для каждого документа, потому что схема была repo-уровня, а не файл-уровня)
def _peek_and_validate_fields(stream_iter, source: SFTSource):
    try:
        first = next(stream_iter)
    except StopIteration:
        raise ValueError(f"{source.dataset_name}: источник пуст (ни одного документа)")

    available = set(first.keys())

    if source.messages_field is not None:
        required = {source.messages_field}
    else:
        required = {source.instruction_field, source.output_field}
        if source.input_field:
            required.add(source.input_field)
        if source.system_field:
            required.add(source.system_field)
    if source.filter_field is not None:
        required.add(source.filter_field)

    missing = required - available
    if missing:
        raise ValueError(
            f"{source.dataset_name}: поля {sorted(missing)} не найдены в данных. "
            f"Реально доступные поля: {sorted(available)}. "
            f"Поправь messages_field (или instruction_field/output_field/input_field/system_field) в data_config.yaml"
        )

    return chain([first], stream_iter)


_VALID_ROLES = {"system", "user", "assistant"}


# Нормализует документ в {"messages": [...]} независимо от исходного формата —
# дальше по пайплайну (форматирование, маскирование) работаем только с этим
# единым представлением, не заботясь о том, из какого источника пришёл пример
def _extract_sft_example(doc: dict, source: SFTSource) -> dict | None:
    if source.messages_field is not None:
        raw_messages = doc.get(source.messages_field)
        if not raw_messages:
            return None

        messages = []
        for m in raw_messages:
            role = m.get("role")
            content = str(m.get("content") or "").strip()
            if role not in _VALID_ROLES or not content:
                continue
            messages.append({"role": role, "content": content})

        has_user = any(m["role"] == "user" for m in messages)
        has_assistant = any(m["role"] == "assistant" for m in messages)
        if not (has_user and has_assistant):
            return None

        return {"messages": messages}

    # Формат Б: instruction/output -> messages из одного user+assistant хода
    instruction = str(doc.get(source.instruction_field) or "").strip()
    output = str(doc.get(source.output_field) or "").strip()
    if not instruction or not output:
        return None

    user_content = instruction
    if source.input_field and doc.get(source.input_field):
        user_content += "\n" + str(doc[source.input_field]).strip()

    messages = []
    if source.system_field and doc.get(source.system_field):
        messages.append({"role": "system", "content": str(doc[source.system_field]).strip()})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output})

    return {"messages": messages}


def _consume(iterator, n: int) -> None:
    deque(islice(iterator, n), maxlen=0)


def _write_sft_shards(valid_iter, save_dir: Path, max_docs: int, rows_per_shard: int) -> int:
    shard_index = 0
    total_written = 0
    while total_written < max_docs:
        take = min(rows_per_shard, max_docs - total_written)
        batch = list(islice(valid_iter, take))
        if not batch:
            break
        table = pa.Table.from_pylist(batch)
        pq.write_table(table, save_dir / f"part_{shard_index:04d}.parquet")
        total_written += len(batch)
        shard_index += 1
    return total_written


# Кеширование одного SFT-источника. Тот же val-first порядок и тот же
# непрерывный итератор (train и val из одного стрима подряд), что и в
# src/data/dataset.py — те же причины: val не должен быть подмножеством train,
# и не должен голодать, если источник физически меньше, чем cache_docs_per_source
def cache_sft_source_to_disk(
    source: SFTSource,
    data_dir: Path,
    train_docs: int,
    val_docs: int,
    rows_per_shard: int = 20_000,
) -> tuple[Path, Path]:
    train_dir = _local_dir_for_sft_source(source, data_dir, "train")
    val_dir = _local_dir_for_sft_source(source, data_dir, "val")
    train_marker = train_dir / "_SUCCESS"
    val_marker = val_dir / "_SUCCESS"

    if train_marker.exists() and val_marker.exists():
        return train_dir, val_dir

    raw_stream = load_dataset(path=source.dataset_name, name=source.subset, split=source.split, streaming=True)
    stream_iter = iter(raw_stream)
    stream_iter = _peek_and_validate_fields(stream_iter, source)

    def valid_examples():
        for doc in stream_iter:
            if source.filter_field is not None and doc.get(source.filter_field) not in source.filter_values:
                continue
            ex = _extract_sft_example(doc, source)
            if ex is not None:
                yield ex

    valid_iter = valid_examples()

    if val_marker.exists():
        _consume(valid_iter, val_docs)
    else:
        val_dir.mkdir(parents=True, exist_ok=True)
        for stale in val_dir.glob("part_*.parquet"):
            stale.unlink()
        written = _write_sft_shards(valid_iter, val_dir, val_docs, rows_per_shard)
        val_marker.write_text(f"docs={written}\n")
        if written < val_docs:
            print(f"⚠️  {source.dataset_name}: источник закончился раньше — val получил {written}/{val_docs}")

    if not train_marker.exists():
        train_dir.mkdir(parents=True, exist_ok=True)
        for stale in train_dir.glob("part_*.parquet"):
            stale.unlink()
        written = _write_sft_shards(valid_iter, train_dir, train_docs, rows_per_shard)
        train_marker.write_text(f"docs={written}\n")
        if written < train_docs:
            print(f"⚠️  {source.dataset_name}: источник закончился раньше — train получил {written}/{train_docs} "
                  f"(валидных non-empty примеров меньше, чем запрошено — проверь качество источника)")

    return train_dir, val_dir


def cache_sft_data(cfg: SFTData, data_dir: Path) -> None:
    for source in cfg.sources:
        train_docs = int(cfg.cache_docs_per_source * (1 - cfg.val_split_ratio))
        val_docs = int(cfg.cache_docs_per_source * cfg.val_split_ratio)
        cache_sft_source_to_disk(source, data_dir, train_docs, val_docs)


def load_local_sft_stream(shard_dir: Path) -> IterableDataset:
    shard_files = sorted(str(p) for p in shard_dir.glob("part_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"Нет шардов в {shard_dir}")
    return load_dataset("parquet", data_files=shard_files, split="train", streaming=True)


def build_sft_mix_from_disk(cfg: SFTData, data_dir: Path, split: str = "train") -> IterableDataset:
    streams = []
    weights = []
    for source in cfg.sources:
        shard_dir = _local_dir_for_sft_source(source, data_dir, split)
        if not shard_dir.exists() or not any(shard_dir.glob("part_*.parquet")):
            print(f"⚠️ Предупреждение: {shard_dir} не найден, пропускаем")
            continue
        streams.append(load_local_sft_stream(shard_dir))
        weights.append(source.weight)

    if not streams:
        raise FileNotFoundError(f"Нет SFT-данных для split={split}")

    total = sum(weights)
    probabilities = [w / total for w in weights]
    return interleave_datasets(streams, probabilities=probabilities, seed=cfg.seed, stopping_strategy="all_exhausted")
