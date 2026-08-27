import torch
import torch.nn as nn
import torch.nn.functional as F


# Класс MLP
class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        # Слои
        self.gate = nn.Linear(hidden_size, intermediate_size)
        self.up = nn.Linear(hidden_size, intermediate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)


    # Прямой проход
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
