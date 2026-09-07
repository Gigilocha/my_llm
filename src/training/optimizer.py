import torch
import torch.nn as nn

# Разделение параметров: Muon — только для 2D-матриц скрытых линейных слоёв
def split_params(model: nn.Module) -> tuple[list, list]:
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1D параметры, эмбеддинги, нормы, биасы и lm_head уходят в AdamW
        is_bias_or_norm = "norm" in name or "bias" in name
        is_embed_or_head = "embed" in name or "lm_head" in name
        
        if param.ndim >= 2 and not is_bias_or_norm and not is_embed_or_head:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


# Ортогонализация через итерацию Ньютона-Шульца
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315

    # Вычисления производятся в bfloat16/float32
    X = G.to(dtype=torch.bfloat16 if G.is_cuda else torch.float32)
    X = X / (X.norm() + eps)

    # Гарантируем, что X "широкая" или "квадратная" для стабильности
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


# Оптимизатор Muon
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
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov, 
            weight_decay=weight_decay, 
            ns_steps=ns_steps
        )
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

                # Decoupled weight decay
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                # Nesterov momentum
                update_grad = grad.add(buf, alpha=momentum) if group["nesterov"] else buf

                # Ортогонализация направления
                u = zeropower_via_newtonschulz5(update_grad, steps=ns_steps)

                # Корректный масштаб с учетом формы матрицы
                scale = max(1.0, (p.size(0) / p.size(1)) ** 0.5)
                p.data.add_(u, alpha=-lr * scale)

        return loss


# Единая обёртка Muon + AdamW
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


# Сборка оптимизатора
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
        fused=torch.cuda.is_available(),
    )

    # Сохраняем base_lr для масштабирования в расписании
    for group in muon_opt.param_groups:
        group["base_lr"] = muon_lr
    for group in adamw_opt.param_groups:
        group["base_lr"] = adamw_lr

    return CombinedOptimizer([muon_opt, adamw_opt])