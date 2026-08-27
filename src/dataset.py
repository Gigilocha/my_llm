from pathlib import Path
from typing import Iterator
from itertools import islice
from datasets import load_dataset, Dataset
from datasets import interleave_datasets, IterableDataset

from src.common.config import DataSource, PretrainData


# Запуск одного стрима данных из сети
def load_source_stream(name: str, subset: str | None, split: str = "train") -> IterableDataset:
    ds = load_dataset(path=name, name=subset, split=split, streaming=True)
    return ds


# Функция построения безопасного имени файла
def _safe_filename(source: DataSource) -> str:
    name = source.dataset_name.replace("/", "_")
    if source.subset:
        name = f"{name}__{source.subset.replace('/', '_')}"
    return f"{name}.parquet"


# Кеширование одного источника на диск (Parquet)
def cache_source_to_disk(source: DataSource, language: str, data_dir: Path, max_docs: int) -> Path:
    save_path = data_dir / language / _safe_filename(source)
    if save_path.exists():
        return save_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    stream = load_source_stream(source.dataset_name, source.subset, source.split)
    docs = list(islice(stream, max_docs))

    from datasets import Dataset
    Dataset.from_list(docs).to_parquet(save_path)
    return save_path


# Кеширование всех источников
def cache_pretrain_data(cfg: PretrainData, data_dir: Path, max_docs: int) -> None:
    for language, sources in [("rus", cfg.rus_sources), ("en", cfg.en_sources), ("code", cfg.code_sources)]:
        for source in sources:
            cache_source_to_disk(source, language, data_dir, max_docs)


# Чтение уже закешированного источника с диска
def load_local_stream(path: Path) -> IterableDataset:
    return load_dataset("parquet", data_files=str(path), split="train", streaming=True)


# Смешать источники одного языка по весам
def build_language_mix(sources: list[DataSource], seed: int) -> IterableDataset:
    streams = []
    weights = []

    for source in sources:
        stream = load_source_stream(source.dataset_name, source.subset, source.split)
        streams.append(stream)
        weights.append(source.weight)

    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    mixed = interleave_datasets(
        streams,
        probabilities=probabilities,
        seed=seed,
    )
    return mixed


# Сборка микса для Pre-training из разных языков
def build_pretrain_mix(cfg: PretrainData) -> IterableDataset:
    rus_mix = build_language_mix(cfg.rus_sources, cfg.seed)
    en_mix = build_language_mix(cfg.en_sources, cfg.seed)
    code_mix = build_language_mix(cfg.code_sources, cfg.seed)

    total_quantity = cfg.rus_quantity + cfg.en_quantity + cfg.code_quantity
    probabilities = [
        cfg.rus_quantity / total_quantity,
        cfg.en_quantity / total_quantity,
        cfg.code_quantity / total_quantity,
    ]

    mixed = interleave_datasets(
        [rus_mix, en_mix, code_mix],
        probabilities=probabilities,
        seed=cfg.seed,
    )
    return mixed


# Конвертация в текст
def extract_texts(dataset: IterableDataset) -> Iterator[str]:
    for doc in dataset:
        text = (doc.get("text") or doc.get("content") or "").strip()
        if text:
            yield text