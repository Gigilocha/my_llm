import torch
import torch.nn as nn


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


# Создание оптимизатора AdamW
def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Оптимизатор Muon

