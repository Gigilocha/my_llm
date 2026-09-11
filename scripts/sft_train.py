from pathlib import Path
from transformers import PreTrainedTokenizerFast
import torch
import mlflow
import os
import time

from src.common.mlflow import (
    log_step, log_eval, log_memory, log_speed,
    log_gradients, log_model_params
)
from src.common.config import get_config, PROJECT_ROOT
from src.common.device import resolve_device, get_device_info
from src.common.logger import setup_logger
from src.data.sft_dataset import cache_sft_data, build_sft_mix_from_disk
from src.data.sft_format import sft_examples_to_tokenized
from src.data.sft_dataloader import collate_sft_batch
from src.model.transformer import Transformer
from src.training.sft_train_step import sft_train_step, sft_eval_step
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


# Бесконечный поток токенизированных (input_ids, labels) SFT-примеров —
# та же идея, что create_cycling_pretrain_dataloader: пересоздаём стрим заново,
# когда один проход по SFT-корпусу заканчивается, вместо падения на пустом батче.
# Если за целый проход не прошёл токенизацию НИ ОДИН пример (например, max_len
# слишком мал относительно длины примеров — промпт+спецтокены уже превышают его) —
# без этой проверки цикл крутился бы вечно вхолостую, ничего не выдавая и не падая
def cycling_sft_examples(mix_factory, tokenizer, special_tokens, max_len):
    while True:
        examples = mix_factory()
        yielded_any = False
        for item in sft_examples_to_tokenized(examples, tokenizer, special_tokens, max_len):
            yielded_any = True
            yield item
        if not yielded_any:
            raise RuntimeError(
                "Ни один SFT-пример не прошёл токенизацию за целый проход по данным — "
                "вероятно max_len слишком мал относительно длины примеров (промпт со "
                "спецтокенами уже превышает max_len). Проверь training_config.yaml:sft.max_len"
            )


def collect_batch(dataloader, batch_size):
    return [next(dataloader) for _ in range(batch_size)]


def main():
    config = get_config()
    device = resolve_device(config.env.device)
    logger = setup_logger(__name__)
    logger.info(f"Устройство: {get_device_info(device)}")

    data_dir = PROJECT_ROOT / config.env.data_dir
    sft_config = config.training.sft
    sft_data_cfg = config.data.sft_data

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_dir = PROJECT_ROOT / "outputs" / "mlflow"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlflow_dir}/mlruns")
    mlflow.start_run(run_name="sft")
    mlflow.set_tags({"phase": "sft", "device": device})
    mlflow.log_params(sft_config.model_dump())

    # Кеширование SFT-данных (если ещё не закешированы — см. sft_dataset.py,
    # та же защита от утечки train/val и от истощения источника, что в pretrain)
    logger.info("Кеширование SFT-данных...")
    cache_sft_data(sft_data_cfg, data_dir)

    tokenizer_path = PROJECT_ROOT / config.env.outputs_dir / "tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    special_tokens = config.tokenizer.special_tokens

    model = build_model(config).to(device)

    # Стартуем с последнего pretrain-чекпоинта — SFT это дообучение, не обучение
    # с нуля. optimizer=None: веса грузим, а optimizer-состояние pretrain (Muon/
    # AdamW momentum-буферы) для SFT не нужно — оптимизатор SFT собираем заново
    pretrain_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    pretrain_step = find_latest_checkpoint(pretrain_checkpoints_dir)
    if pretrain_step is None:
        raise FileNotFoundError(f"Нет pretrain-чекпоинтов в {pretrain_checkpoints_dir} — сначала обучи базовую модель")
    load_checkpoint(checkpoint_dir=pretrain_checkpoints_dir, step=pretrain_step, model=model)
    logger.info(f"Загружены веса pretrain-чекпоинта (шаг {pretrain_step})")

    optimizer = build_optimizer(
        model,
        adamw_lr=sft_config.learning_rate,
        adamw_weight_decay=sft_config.weight_decay,
        muon_lr=sft_config.muon_learning_rate,
        muon_weight_decay=sft_config.muon_weight_decay,
        muon_momentum=sft_config.muon_momentum,
    )

    sft_checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints_sft"
    latest_sft_step = find_latest_checkpoint(sft_checkpoints_dir)
    if latest_sft_step is not None:
        load_checkpoint(checkpoint_dir=sft_checkpoints_dir, step=latest_sft_step, model=model, optimizer=optimizer)
        start_step = latest_sft_step + 1
        logger.info(f"Восстановлено SFT-обучение с шага {start_step}")
    else:
        start_step = 0

    compiled_model = torch.compile(model) if device == "cuda" else model

    train_mix_factory = lambda: build_sft_mix_from_disk(sft_data_cfg, data_dir, split="train")
    train_dataloader = cycling_sft_examples(train_mix_factory, tokenizer, special_tokens, sft_config.max_len)

    val_mix_factory = lambda: build_sft_mix_from_disk(sft_data_cfg, data_dir, split="val")
    val_dataloader = cycling_sft_examples(val_mix_factory, tokenizer, special_tokens, sft_config.max_len)

    grad_accum_steps = sft_config.gradient_accumulation_steps
    log_model_params(model, logger)
    log_memory(logger, tag="start")

    for step in range(start_step, sft_config.max_steps):
        lr = get_lr(
            step=step,
            warmup_steps=sft_config.warmup_steps,
            max_steps=sft_config.max_steps,
            learning_rate=sft_config.learning_rate,
            min_learning_rate=sft_config.min_learning_rate,
        )
        lr_scale = lr / sft_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["base_lr"] * lr_scale

        optimizer.zero_grad()
        accumulated_loss = 0.0
        start_time = time.time()

        for _ in range(grad_accum_steps):
            examples = collect_batch(train_dataloader, sft_config.batch_size)
            input_ids, labels, attention_mask = collate_sft_batch(examples, tokenizer.pad_token_id)
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)

            loss = sft_train_step(compiled_model, input_ids, labels, attention_mask, grad_accum_steps)
            accumulated_loss += loss

        avg_loss = accumulated_loss / grad_accum_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=sft_config.grad_clip_norm)
        optimizer.step()
        elapsed = time.time() - start_time

        log_step(logger, step, avg_loss, lr, grad_norm.item())
        log_speed(logger, step, sft_config.batch_size, sft_config.max_len, elapsed, grad_accum_steps)
        log_memory(logger, step, tag="training")
        log_gradients(model, step, logger)

        if step % sft_config.checkpoint_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, avg_loss, sft_checkpoints_dir)
            logger.info(f"SFT-чекпоинт сохранён на шаге {step}")

        if step % sft_config.eval_interval == 0 and step > 0:
            val_examples = collect_batch(val_dataloader, sft_config.batch_size)
            v_ids, v_labels, v_mask = collate_sft_batch(val_examples, tokenizer.pad_token_id)
            v_ids, v_labels, v_mask = v_ids.to(device), v_labels.to(device), v_mask.to(device)
            val_loss = sft_eval_step(compiled_model, v_ids, v_labels, v_mask)
            log_eval(logger, step, val_loss)
            mlflow.log_metric("val_loss", val_loss, step=step)

    mlflow.end_run()


if __name__ == "__main__":
    main()
