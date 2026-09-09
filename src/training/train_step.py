import torch
import torch.nn as nn
import torch.nn.functional as F


# Функция шага тренеровки.
# grad_accum_steps > 1: loss делится перед backward, чтобы градиенты нескольких
# микро-батчей усреднялись, а не суммировались — иначе эффективный lr незаметно
# вырастает в grad_accum_steps раз. Возвращается немасштабированный loss (для логов).
#
# autocast(bf16): matmul/attention считаются в bf16 (в 2 раза меньше памяти на
# активации, быстрее на тензорных ядрах), веса и градиенты остаются в fp32 —
# в отличие от fp16, bf16 не требует GradScaler (тот же диапазон экспоненты,
# что и fp32, переполнение практически не грозит)
def train_step(model: nn.Module, batch: torch.Tensor, grad_accum_steps: int = 1) -> float:
    inputs = batch[:, :-1] # Получение данных для последующего предсказания
    labels = batch[:, 1:] # Получение меток для предсказания

    device_type = "cuda" if batch.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=batch.is_cuda):
        # Прямой проход по модели
        logits = model(inputs)

        # Выпрямление данных
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.reshape(-1)

        # Вычисление ошибки
        loss = F.cross_entropy(logits, labels)

    # backward — вне autocast (так и должно быть: PyTorch сам решает, в какой
    # точности считать градиенты по сохранённым во время forward метаданным)
    (loss / grad_accum_steps).backward()

    return loss.item()











    