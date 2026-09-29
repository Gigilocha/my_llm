import argparse
import random

import pyarrow.parquet as pq

from src.common.config import get_config, PROJECT_ROOT
from src.common.logger import setup_logger
from src.data.dataset import dpo_shard_dir


"""
Проверка закешированных DPO-данных до обучения: количество train/val на
источник + случайные примеры (prompt/chosen/rejected) для чтения глазами.
Тот же принцип, что sft_data_eval.py — target_share показан рядом с
фактическим weight, weight ещё НЕ скорректирован калибровкой (она происходит
в момент сборки микса, build_dpo_mix_from_disk, не здесь).
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
    parser = argparse.ArgumentParser(description="Проверка закешированных DPO-данных: количество + примеры глазами")
    parser.add_argument("--samples-per-source", type=int, default=5, help="Сколько случайных примеров показать на источник")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-chars", type=int, default=300, help="Сколько символов показывать в превью каждого поля")
    args = parser.parse_args()

    config = get_config()
    logger = setup_logger(__name__)
    data_dir = PROJECT_ROOT / config.env.data_dir
    rng = random.Random(args.seed)

    for source in config.data.rlft_data.sources:
        share_info = f"target_share={source.target_share}" if source.target_share is not None else "target_share не задан"
        logger.info(f"\n{'='*60}\nИсточник: {source.dataset_name} (weight={source.weight}, {share_info})")

        for split in ("train", "val"):
            shard_dir = dpo_shard_dir(source, data_dir, split)
            marker = shard_dir / "_SUCCESS"
            if not marker.exists():
                logger.info(f"  {split}: НЕ ЗАКЕШИРОВАН")
                continue

            doc_count = marker.read_text().strip()
            n_shards = len(list(shard_dir.glob("part_*.parquet")))
            logger.info(f"  {split}: {doc_count}, шардов: {n_shards}")

        train_dir = dpo_shard_dir(source, data_dir, "train")
        if train_dir.exists():
            examples = read_examples(train_dir, args.samples_per_source)
            if examples:
                sample = rng.sample(examples, min(args.samples_per_source, len(examples)))
                for i, ex in enumerate(sample):
                    prompt_preview = " / ".join(m["content"][:args.preview_chars] for m in ex["messages"])
                    logger.info(f"\n  --- Пример {i+1} ---")
                    logger.info(f"  [prompt]: {prompt_preview}")
                    logger.info(f"  [chosen]: {ex['chosen'][:args.preview_chars]}")
                    logger.info(f"  [rejected]: {ex['rejected'][:args.preview_chars]}")


if __name__ == "__main__":
    main()