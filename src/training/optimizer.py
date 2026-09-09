import torch
import torch.nn as nn


"""
Заметка:
Muon — оптимизатор для 2D-матриц (веса линейных слоёв), не для эмбеддингов/lm_head/норм.
Идея: обычный SGD-момент даёт направление обновления, но оно может быть "вытянутым"
вдоль немногих сингулярных векторов (плохо обусловлено). Muon ортогонализирует это
направление через итерацию Ньютона-Шульца (без явного дорогого SVD) — получается
обновление с примерно равным вкладом по всем направлениям, что на матрицах трансформера
даёт более быструю и стабильную сходимость, чем AdamW.
Эмбеддинги/lm_head/нормы/bias — 1D или "по строкам не как матрица преобразования",
для них ортогонализация не имеет смысла, поэтому они остаются на AdamW.
Референс: Keller Jordan, "Muon: An optimizer for hidden layers in neural networks".
"""


# Разделение параметров
def split_params(model: nn.Module) -> tuple[list, list]:
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if param.ndim >= 2 and "embedding" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


# Ортогонализация матрицы через итерацию Ньютона-Шульца (5 шагов по умолчанию).
# Быстрее и дешевле честного SVD, даёт приближение к ближайшей ортогональной матрице.
# Коэффициенты (a, b, c) — стандартные квинтические коэффициенты из референса Muon,
# подобранные так, чтобы итерация сходилась за минимум шагов без переполнения.
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.to(dtype=torch.bfloat16 if G.is_cuda else torch.float32)
    X = X / (X.norm() + eps)

    # Итерация устойчивее для "высоких" матриц (строк больше, чем столбцов) —
    # если наоборот, транспонируем на время итерации и обратно в конце
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(dtype=G.dtype)


# Оптимизатор Muon: момент + ортогонализация направления обновления
class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Decoupled weight decay (как в AdamW) — не мешается с градиентом до момента
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                # Nesterov: используем "предсказанный" момент, а не сам буфер
                update_grad = grad.add(buf, alpha=momentum) if group["nesterov"] else buf

                u = zeropower_via_newtonschulz5(update_grad, steps=ns_steps)

                # Масштаб под форму матрицы: без этого шаг обновления по норме
                # непропорционально различался бы между "широкими" и "высокими" весами
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.data.add_(u, alpha=-lr * scale)

        return loss


# Единая обёртка над Muon (для матриц) + AdamW (для эмбеддингов/lm_head/норм/bias),
# чтобы для остального кода (train loop, checkpoint save/load) это выглядело
# как один обычный torch.optim.Optimizer — не нужно менять base_train.py/checkpoint.py
class CombinedOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict:
        return {f"optimizer_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict: dict) -> None:
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict[f"optimizer_{i}"])


# Создание оптимизатора: Muon для матриц трансформера, AdamW для остального.
# У групп разный базовый lr (Muon обычно на порядок больше AdamW) — base_lr сохраняем
# в group, чтобы LR-scheduler мог применять один и тот же прогресс warmup/decay
# к обеим группам пропорционально их пиковым значениям, а не переписывать их одним числом
def build_optimizer(
    model: nn.Module,
    adamw_lr: float,
    adamw_weight_decay: float,
    muon_lr: float = 0.02,
    muon_weight_decay: float = 0.0,
    muon_momentum: float = 0.95,
) -> CombinedOptimizer:
    muon_params, adamw_params = split_params(model)

    muon_opt = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=muon_weight_decay,
    )
    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        weight_decay=adamw_weight_decay,
        # fused: один CUDA-кернел на все параметры вместо Python-цикла по ним.
        # Работает только на CUDA-тензорах — на CPU (тесты, отладка) тихо не включаем
        fused=torch.cuda.is_available(),
    )

    # base_lr — точка отсчёта для масштабирования расписанием (см. base_train.py)
    for group in muon_opt.param_groups:
        group["base_lr"] = muon_lr
    for group in adamw_opt.param_groups:
        group["base_lr"] = adamw_lr

    return CombinedOptimizer([muon_opt, adamw_opt])
