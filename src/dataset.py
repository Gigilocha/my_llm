from typing import Iterator
from datasets import load_dataset
from datasets import interleave_datasets, IterableDataset

from src.common.config import DataSource, PretrainData


# Запуск одного стрима данных
def load_source_stream(name: str, subset: str | None, split: str = "train") -> IterableDataset:
    ds = load_dataset(path=name, name=subset, split=split, streaming=True)
    return ds


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