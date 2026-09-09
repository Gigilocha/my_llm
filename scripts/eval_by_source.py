import argparse
import math

from transformers import PreTrainedTokenizerFast

from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.dataset import build_language_mix_from_disk, extract_texts
from src.data.dataloader import create_cycling_pretrain_dataloader, collate_pretrain_batch
from src.model.transformer import Transformer
from src.training.checkpoint import find_latest_checkpoint, load_checkpoint
from src.training.eval_step import eval_step


"""
base_eval.py даёт val_loss по ЯЗЫКУ целиком (rus = fineweb-2 + wikipedia вместе).
Этого недостаточно, когда вопрос "какой ИЗ ДВУХ источников внутри rus хуже" —
нужен loss по каждому источнику отдельно. Один источник = один DataSource,
поэтому build_language_mix_from_disk([source], ...) с списком из одного элемента
даёт честный поток только по нему, без смешивания с остальными источниками языка.
"""


def build_model(config) -> Transformer:
    return Transformer(
        vocab_size=config.model.model.vocab_size,
        num_layer=config.model.model.num_layers,
        hidden_size=config.model.model.hidden_size,
        head_dim=config.model.attention.head_dim,
        num_heads=config.model.attention.num_heads,
        num_kv_heads=config.model.attention.num_kv_heads,
        use_qk_norm=config.model.attention.use_qk_norm,
        qk_norm_eps=config.model.attention.qk_norm_eps,
        rope_theta=config.model.attention.rope_theta,
        max_position_embeddings=config.model.model.max_position_embeddings,
        intermediate_size=config.model.mlp.intermediate_size,
        norm_eps=config.model.model.norm_eps,
    )


def main():
    parser = argparse.ArgumentParser(description="val_loss отдельно по каждому источнику внутри языка")
    parser.add_argument("--language", type=str, default="rus", choices=["rus", "en", "code"])
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num-eval-batches", type=int, default=50)
    args = parser.parse_args()

    config = get_config()
    device = resolve_device(config.env.device)
    logger = setup_logger(__name__)
    logger.info(f"Устройство: {get_device_info(device)}")

    data_dir = PROJECT_ROOT / config.env.data_dir
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    step = args.step if args.step is not None else find_latest_checkpoint(checkpoints_dir)
    if step is None:
        raise FileNotFoundError(f"Нет чекпоинтов в {checkpoints_dir}")
    logger.info(f"Чекпоинт: шаг {step}")

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    model = build_model(config).to(device)
    load_checkpoint(checkpoint_dir=checkpoints_dir, step=step, model=model)
    model.eval()

    pretrain_config = config.training.pre_training
    sources = getattr(config.data.pre_training_data, f"{args.language}_sources")

    logger.info(f"Источников в {args.language}: {len(sources)}")
    for source in sources:
        label = source.dataset_name + (f"/{source.subset}" if source.subset else "")

        text_factory = lambda: extract_texts(
            build_language_mix_from_disk([source], "pretrain", args.language, data_dir, config.data.pre_training_data.seed, split="val")
        )
        dataloader = create_cycling_pretrain_dataloader(text_factory, tokenizer, pretrain_config.max_len)

        losses = []
        for _ in range(args.num_eval_batches):
            batch = collate_pretrain_batch(dataloader, pretrain_config.batch_size).to(device)
            losses.append(eval_step(model, batch))

        avg_loss = sum(losses) / len(losses)
        logger.info(f"  {label}: val_loss={avg_loss:.4f}, perplexity={math.exp(avg_loss):.2f}")


if __name__ == "__main__":
    main()
