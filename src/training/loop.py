import time
from typing import Callable

import torch

from src.training.lr_schedule import get_lr


"""
Общий скелет цикла обучения для pretrain/SFT/DPO. Разница между стадиями —
ЧТО происходит на одном микрошаге (train_step/sft_train_step/dpo_train_step —
разные сигнатуры, DPO дополнительно требует reference-модель) — остаётся
явной в каждом train-скрипте через micro_step(), а не спрятана сюда через
if stage == ... (это была бы не дедупликация, а её иллюзия — тот же код на
три пути, просто менее читаемый). Общее — сам скелет: накопление
grad_accum_steps микрошагов, lr-schedule, clip, шаг оптимизатора, диспетчер
checkpoint/eval/логирования по интервалам.

Логирование (log_speed/log_memory/log_gradients) везде гейтится ОДНИМ и тем
же interval_steps из monitoring_config.yaml — раньше SFT/DPO логировали это
на каждом шаге, а pretrain — по интервалу; расхождение было случайным (тот
же конфиг использовался наполовину), не осознанным решением. Единственное,
что логируется каждый шаг всегда — loss/lr/grad_norm (log_step) — это
дёшево (три числа), в отличие от gradient-статистики по всем параметрам.
"""


def run_training_loop(
    *,
    start_step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
    grad_clip_norm: float,
    checkpoint_interval: int,
    eval_interval: int,
    grad_accum_steps: int,
    optimizer,
    model_for_clip: torch.nn.Module,
    # Один forward+backward -> (loss, доп. метрики стадии — например accuracy/
    # reward_margin у DPO; {} у pretrain/SFT, где доп. метрик нет)
    micro_step: Callable[[], tuple[float, dict]],
    # Вызывается КАЖДЫЙ шаг с усреднёнными по grad_accum_steps значениями —
    # сам решает, что логировать каждый шаг, а что по интервалу (через свой
    # monitoring_config), это ответственность вызывающего скрипта, не цикла
    on_step: Callable[[int, float, float, float, float, dict], None],
    checkpoint_fn: Callable[[int, float], None],
    eval_fn: Callable[[int], None] | None = None,
) -> None:
    for step in range(start_step, max_steps):
        start_time = time.time()

        lr = get_lr(
            step=step, warmup_steps=warmup_steps, max_steps=max_steps,
            learning_rate=learning_rate, min_learning_rate=min_learning_rate,
        )
        # lr_scale — прогресс по расписанию (0..1 с учётом warmup/cosine), общий
        # для всех групп; base_lr у каждой группы свой (Muon намного больше AdamW),
        # так что абсолютный lr у групп разный, а форма расписания — одна и та же
        lr_scale = lr / learning_rate
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_scale

        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_metrics: dict[str, float] = {}

        for _ in range(grad_accum_steps):
            loss, metrics = micro_step()
            accumulated_loss += loss
            for key, value in metrics.items():
                accumulated_metrics[key] = accumulated_metrics.get(key, 0.0) + value

        avg_loss = accumulated_loss / grad_accum_steps
        avg_metrics = {key: value / grad_accum_steps for key, value in accumulated_metrics.items()}
        grad_norm = torch.nn.utils.clip_grad_norm_(model_for_clip.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        elapsed = time.time() - start_time

        on_step(step, avg_loss, lr, grad_norm.item(), elapsed, avg_metrics)

        if step % checkpoint_interval == 0 and step > 0:
            checkpoint_fn(step, avg_loss)

        if eval_fn is not None and step % eval_interval == 0 and step > 0:
            eval_fn(step)