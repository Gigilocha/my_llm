import copy
import os
import time

import mlflow
import torch
from transformers import PreTrainedTokenizerFast

from src.common.mlflow import log_step, log_eval, log_memory, log_speed, log_gradients, log_model_params
from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.dpo_dataset import cache_dpo_data, build_dpo_mix_from_disk
from src.data.dpo_format import dpo_examples_to_tokenized
from src.data.sft_dataloader import collate_sft_batch
from src.model.transformer import Transformer
from src.training.dpo_train_step import dpo_train_step, dpo_eval_step
from src.training.optimizer import build_optimizer
from src.training.lr_schedule import get_lr
from src.training.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


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


def cycling_dpo_examples(mix_factory, tokenizer, special_tokens, max_len):
    while True:
        yielded_any = False
        for item in dpo_examples_to_tokenized(mix_factory(), tokenizer, special_tokens, max_len):
            yielded_any = True
            yield item
        if not yielded_any:
            raise RuntimeError(
                "Ни одна DPO-пара не прошла токенизацию за целый проход — вероятно "
                "max_len слишком мал относительно длины примеров. Проверь training_config.yaml:rlft.max_len"
            )


def collect_pair_batch(dataloader, batch_size, pad_token_id):
    pairs = [next(dataloader) for _ in range(batch_size)]
    chosen = collate_sft_batch([c for c, _ in pairs], pad_token_id)
    rejected = collate_sft_batch([r for _, r in pairs], pad_token_id)
    return chosen, rejected


def main():
    config = get_config()
    device = resolve_device(config.env.device)
    logger = setup_logger(__name__)
    logger.info(f"Устройство: {get_device_info(device)}")

    data_dir = PROJECT_ROOT / config.env.data_dir
    rlft_config = config.training.rlft
    dpo_data_cfg = config.data.rlft_data

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_dir = PROJECT_ROOT / "outputs" / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlflow_dir}/mlruns")
    mlflow.start_run(run_name="rlft_dpo")
    mlflow.set_tags({"phase": "rlft", "method": "dpo", "device": device})
    mlflow.log_params(rlft_config.model_dump())

    logger.info("Кеширование DPO-данных...")
    cache_dpo_data(dpo_data_cfg, data_dir)

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    special_tokens = config.tokenizer.special_tokens

    # Политика стартует с SFT-чекпоинта — DPO это выравнивание уже обученной
    # отвечать модели, не обучение с нуля
    sft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_sft"
    sft_step = find_latest_checkpoint(sft_checkpoints_dir)
    if sft_step is None:
        raise FileNotFoundError(f"Нет SFT-чекпоинтов в {sft_checkpoints_dir} — сначала запусти base_sft.py")

    policy_model = build_model(config).to(device)
    load_checkpoint(checkpoint_dir=sft_checkpoints_dir, step=sft_step, model=policy_model)
    logger.info(f"Политика инициализирована из SFT-чекпоинта (шаг {sft_step})")

    # Reference-модель — замороженная КОПИЯ той же стартовой точки. Копируем
    # ДО любых шагов оптимизатора, иначе reference уедет вместе с политикой
    # и регуляризация перестанет работать
    ref_model = copy.deepcopy(policy_model).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = build_optimizer(
        policy_model,
        adamw_lr=rlft_config.learning_rate,
        adamw_weight_decay=rlft_config.weight_decay,
        muon_lr=rlft_config.muon_learning_rate,
        muon_weight_decay=rlft_config.muon_weight_decay,
        muon_momentum=rlft_config.muon_momentum,
    )

    rlft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_rlft"
    latest_rlft_step = find_latest_checkpoint(rlft_checkpoints_dir)
    if latest_rlft_step is not None:
        load_checkpoint(checkpoint_dir=rlft_checkpoints_dir, step=latest_rlft_step,
                        model=policy_model, optimizer=optimizer)
        start_step = latest_rlft_step + 1
        logger.info(f"Восстановлено DPO-обучение с шага {start_step}")
    else:
        start_step = 0

    compiled_policy = torch.compile(policy_model) if device == "cuda" else policy_model
    compiled_ref = torch.compile(ref_model) if device == "cuda" else ref_model

    train_dataloader = cycling_dpo_examples(
        lambda: build_dpo_mix_from_disk(dpo_data_cfg, data_dir, split="train"),
        tokenizer, special_tokens, rlft_config.max_len,
    )
    val_dataloader = cycling_dpo_examples(
        lambda: build_dpo_mix_from_disk(dpo_data_cfg, data_dir, split="val"),
        tokenizer, special_tokens, rlft_config.max_len,
    )

    grad_accum_steps = rlft_config.gradient_accumulation_steps
    log_model_params(policy_model, logger)
    log_memory(logger, tag="start")

    for step in range(start_step, rlft_config.max_steps):
        lr = get_lr(
            step=step,
            warmup_steps=rlft_config.warmup_steps,
            max_steps=rlft_config.max_steps,
            learning_rate=rlft_config.learning_rate,
            min_learning_rate=rlft_config.min_learning_rate,
        )
        lr_scale = lr / rlft_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["base_lr"] * lr_scale

        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_metrics = {"accuracy": 0.0, "reward_margin": 0.0}
        start_time = time.time()

        for _ in range(grad_accum_steps):
            (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = collect_pair_batch(
                train_dataloader, rlft_config.batch_size, tokenizer.pad_token_id
            )
            c_ids, c_labels, c_mask = c_ids.to(device), c_labels.to(device), c_mask.to(device)
            r_ids, r_labels, r_mask = r_ids.to(device), r_labels.to(device), r_mask.to(device)

            loss, metrics = dpo_train_step(
                compiled_policy, compiled_ref,
                c_ids, c_labels, c_mask, r_ids, r_labels, r_mask,
                beta=rlft_config.beta, grad_accum_steps=grad_accum_steps,
            )
            accumulated_loss += loss
            for key in accumulated_metrics:
                accumulated_metrics[key] += metrics[key]

        avg_loss = accumulated_loss / grad_accum_steps
        avg_metrics = {k: v / grad_accum_steps for k, v in accumulated_metrics.items()}
        grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=rlft_config.grad_clip_norm)
        optimizer.step()
        elapsed = time.time() - start_time

        log_step(logger, step, avg_loss, lr, grad_norm.item())
        logger.info(f"step {step}, accuracy={avg_metrics['accuracy']:.3f}, margin={avg_metrics['reward_margin']:.4f}")
        mlflow.log_metric("dpo_accuracy", avg_metrics["accuracy"], step=step)
        mlflow.log_metric("reward_margin", avg_metrics["reward_margin"], step=step)
        log_speed(logger, step, rlft_config.batch_size, rlft_config.max_len, elapsed, grad_accum_steps)
        log_memory(logger, step, tag="training")
        log_gradients(policy_model, step, logger)

        if step % rlft_config.checkpoint_interval == 0 and step > 0:
            save_checkpoint(policy_model, optimizer, step, avg_loss, rlft_checkpoints_dir)
            logger.info(f"DPO-чекпоинт сохранён на шаге {step}")

        if step % rlft_config.eval_interval == 0 and step > 0:
            (c_ids, c_labels, c_mask), (r_ids, r_labels, r_mask) = collect_pair_batch(
                val_dataloader, rlft_config.batch_size, tokenizer.pad_token_id
            )
            c_ids, c_labels, c_mask = c_ids.to(device), c_labels.to(device), c_mask.to(device)
            r_ids, r_labels, r_mask = r_ids.to(device), r_labels.to(device), r_mask.to(device)
            val_loss, val_acc = dpo_eval_step(
                compiled_policy, compiled_ref,
                c_ids, c_labels, c_mask, r_ids, r_labels, r_mask, beta=rlft_config.beta,
            )
            log_eval(logger, step, val_loss)
            logger.info(f"step {step}, val_accuracy={val_acc:.3f}")
            mlflow.log_metric("val_loss", val_loss, step=step)
            mlflow.log_metric("val_dpo_accuracy", val_acc, step=step)

    mlflow.end_run()


if __name__ == "__main__":
    main()
