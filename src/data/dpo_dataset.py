from pathlib import Path
from itertools import islice, chain
from collections import deque

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, interleave_datasets, IterableDataset

from src.common.config import DPOSource, DPOData


"""
Кеширование preference-данных для DPO. Внутренний формат, к которому всё
нормализуется (единый, независимо от исходной схемы источника):

    {"messages": [...],          # промпт как история диалога (без ответа)
     "chosen": "...",            # предпочтительный ответ, текст
     "rejected": "..."}          # отвергнутый ответ, текст

Та же защита, что в pretrain/SFT: val пишется ПЕРВЫМ из одного непрерывного
итератора (нет утечки train/val и val не голодает, если источник мал),
валидация полей на первом документе (не кешируем молча пустоту).
"""


def _local_dir_for_dpo_source(source: DPOSource, data_dir: Path, split: str) -> Path:
    if source.cache_name:
        name = source.cache_name.replace("/", "_")
    else:
        name = source.dataset_name.replace("/", "_")
        if source.subset:
            name = f"{name}__{source.subset.replace('/', '_')}"
    return data_dir / "rlft" / split / name


def _peek_and_validate_fields(stream_iter, source: DPOSource):
    try:
        first = next(stream_iter)
    except StopIteration:
        raise ValueError(f"{source.dataset_name}: источник пуст (ни одного документа)")

    available = set(first.keys())
    required = {source.prompt_field, source.chosen_field, source.rejected_field}
    if source.filter_field is not None:
        required.add(source.filter_field)

    missing = required - available
    if missing:
        raise ValueError(
            f"{source.dataset_name}: поля {sorted(missing)} не найдены в данных. "
            f"Реально доступные поля: {sorted(available)}. "
            f"Поправь prompt_field/chosen_field/rejected_field в data_config.yaml"
        )

    return chain([first], stream_iter)


_VALID_ROLES = {"system", "user", "assistant"}


# Достаёт текст ответа: из строки как есть, из списка messages — content
# последнего хода assistant (типичная раскладка preference-датасетов)
def _response_text(value, as_messages: bool) -> str:
    if not as_messages or isinstance(value, str):
        return str(value or "").strip()

    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content") or "").strip()
    return ""


# Промпт -> список messages (история диалога без финального ответа)
def _prompt_messages(value, as_messages: bool) -> list[dict]:
    if not as_messages or isinstance(value, str):
        text = str(value or "").strip()
        return [{"role": "user", "content": text}] if text else []

    if isinstance(value, list):
        messages = []
        for message in value:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if role in _VALID_ROLES and content:
                messages.append({"role": role, "content": content})
        # Хвостовой ответ assistant в промпт не входит — он и есть то, что
        # сравнивается через chosen/rejected
        while messages and messages[-1]["role"] == "assistant":
            messages.pop()
        return messages
    return []


def _extract_dpo_example(doc: dict, source: DPOSource) -> dict | None:
    messages = _prompt_messages(doc.get(source.prompt_field), source.as_messages)
    chosen = _response_text(doc.get(source.chosen_field), source.as_messages)
    rejected = _response_text(doc.get(source.rejected_field), source.as_messages)

    if not messages or not chosen or not rejected:
        return None
    # Одинаковые ответы не несут сигнала предпочтения — градиент DPO на такой
    # паре ровно нулевой, смысла хранить нет
    if chosen == rejected:
        return None

    return {"messages": messages, "chosen": chosen, "rejected": rejected}


def _consume(iterator, n: int) -> None:
    deque(islice(iterator, n), maxlen=0)


def _write_shards(valid_iter, save_dir: Path, max_docs: int, rows_per_shard: int) -> int:
    shard_index = 0
    total_written = 0
    while total_written < max_docs:
        take = min(rows_per_shard, max_docs - total_written)
        batch = list(islice(valid_iter, take))
        if not batch:
            break
        pq.write_table(pa.Table.from_pylist(batch), save_dir / f"part_{shard_index:04d}.parquet")
        total_written += len(batch)
        shard_index += 1
    return total_written


def cache_dpo_source_to_disk(
    source: DPOSource,
    data_dir: Path,
    train_docs: int,
    val_docs: int,
    rows_per_shard: int = 20_000,
) -> tuple[Path, Path]:
    train_dir = _local_dir_for_dpo_source(source, data_dir, "train")
    val_dir = _local_dir_for_dpo_source(source, data_dir, "val")
    train_marker = train_dir / "_SUCCESS"
    val_marker = val_dir / "_SUCCESS"

    if train_marker.exists() and val_marker.exists():
        return train_dir, val_dir

    raw_stream = load_dataset(path=source.dataset_name, name=source.subset, split=source.split, streaming=True)
    stream_iter = _peek_and_validate_fields(iter(raw_stream), source)

    def valid_examples():
        for doc in stream_iter:
            if source.filter_field is not None and doc.get(source.filter_field) not in source.filter_values:
                continue
            example = _extract_dpo_example(doc, source)
            if example is not None:
                yield example

    valid_iter = valid_examples()

    if val_marker.exists():
        _consume(valid_iter, val_docs)
    else:
        val_dir.mkdir(parents=True, exist_ok=True)
        for stale in val_dir.glob("part_*.parquet"):
            stale.unlink()
        written = _write_shards(valid_iter, val_dir, val_docs, rows_per_shard)
        val_marker.write_text(f"docs={written}\n")
        if written < val_docs:
            print(f"⚠️  {source.dataset_name}: источник закончился раньше — val получил {written}/{val_docs}")

    if not train_marker.exists():
        train_dir.mkdir(parents=True, exist_ok=True)
        for stale in train_dir.glob("part_*.parquet"):
            stale.unlink()
        written = _write_shards(valid_iter, train_dir, train_docs, rows_per_shard)
        train_marker.write_text(f"docs={written}\n")
        if written < train_docs:
            print(f"⚠️  {source.dataset_name}: источник закончился раньше — train получил {written}/{train_docs}")

    return train_dir, val_dir


def cache_dpo_data(cfg: DPOData, data_dir: Path) -> None:
    for source in cfg.sources:
        train_docs = int(cfg.cache_docs_per_source * (1 - cfg.val_split_ratio))
        val_docs = int(cfg.cache_docs_per_source * cfg.val_split_ratio)
        cache_dpo_source_to_disk(source, data_dir, train_docs, val_docs)


def load_local_dpo_stream(shard_dir: Path) -> IterableDataset:
    shard_files = sorted(str(p) for p in shard_dir.glob("part_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"Нет шардов в {shard_dir}")
    return load_dataset("parquet", data_files=shard_files, split="train", streaming=True)


def build_dpo_mix_from_disk(cfg: DPOData, data_dir: Path, split: str = "train") -> IterableDataset:
    streams, weights = [], []
    for source in cfg.sources:
        shard_dir = _local_dir_for_dpo_source(source, data_dir, split)
        if not shard_dir.exists() or not any(shard_dir.glob("part_*.parquet")):
            print(f"⚠️ Предупреждение: {shard_dir} не найден, пропускаем")
            continue
        streams.append(load_local_dpo_stream(shard_dir))
        weights.append(source.weight)

    if not streams:
        raise FileNotFoundError(f"Нет DPO-данных для split={split}")

    total = sum(weights)
    return interleave_datasets(
        streams, probabilities=[w / total for w in weights], seed=cfg.seed, stopping_strategy="all_exhausted"
    )
