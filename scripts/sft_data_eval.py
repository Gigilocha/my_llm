import argparse
import random

import pyarrow.parquet as pq

from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import sft_shard_dir


"""
Проверка закешированных SFT-данных до того, как тратить время на обучение:
- сколько примеров реально получилось на источник (train/val)
- случайные примеры для чтения глазами — особенно важно для источников с
  непроверенным происхождением
- target_share рядом с фактическим weight: до автоматической калибровки в
  build_sft_mix_from_disk weight — просто исходное значение из конфига, не
  скорректированное под токенную долю; этот скрипт не показывает "правильный"
  weight (это делает калибровка в момент сборки микса), только сырые счётчики
"""


def read_examples(shard_dir, n: int) -> list[dict]:
    shards = sorted(shard_dir.glob("part_*.parquet"))
    if not shards:
        return []
    examples = []
    for shard_path in shards:
        table = pq.read_table(shard_path)
        examples.extend(table.to_pylist())
        if len(examples) >= n * 20:  # не читаем весь корпус ради выборки
            break
    return examples


def main():
    parser = argparse.ArgumentParser(description="Проверка закешированных SFT-данных: количество + примеры глазами")
    parser.add_argument("--samples-per-source", type=int, default=5, help="Сколько случайных примеров показать на источник")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = get_config()
    logger = setup_logger(__name__)
    data_dir = PROJECT_ROOT / config.env.data_dir
    rng = random.Random(args.seed)

    for source in config.data.sft_data.sources:
        share_info = f"target_share={source.target_share}" if source.target_share is not None else "target_share не задан"
        logger.info(f"\n{'='*60}\nИсточник: {source.dataset_name} (weight={source.weight}, {share_info})")

        for split in ("train", "val"):
            shard_dir = sft_shard_dir(source, data_dir, split)
            marker = shard_dir / "_SUCCESS"
            if not marker.exists():
                logger.info(f"  {split}: НЕ ЗАКЕШИРОВАН")
                continue

            doc_count = marker.read_text().strip()
            n_shards = len(list(shard_dir.glob("part_*.parquet")))
            logger.info(f"  {split}: {doc_count}, шардов: {n_shards}")

        train_dir = sft_shard_dir(source, data_dir, "train")
        if train_dir.exists():
            examples = read_examples(train_dir, args.samples_per_source)
            if examples:
                sample = rng.sample(examples, min(args.samples_per_source, len(examples)))
                for i, ex in enumerate(sample):
                    logger.info(f"\n  --- Пример {i+1} ({len(ex['messages'])} ходов) ---")
                    for message in ex["messages"]:
                        content_preview = message["content"][:300]
                        logger.info(f"  [{message['role']}]: {content_preview}")


if __name__ == "__main__":
    main()