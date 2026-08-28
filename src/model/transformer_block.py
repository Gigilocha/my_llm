import torch
import torch.nn as nn

from src.model.mlp import SwiGLU
from src.model.attention import Attention


# Класс одного блока трансформера
class TransformerBlock(nn.Module):
    def __init(self, hidden_size: int, head_dim: int, num_heads: int, num_kv_heads: int, use_qk_norm: bool, 
               qk_norm_eps: float, rope_theta: float, max_position_embeddings: int,
               intermediate_size: int, norm_eps: float):
        super().__init__()

        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attnetion =Attention(
            hidden_size=hidden_size, head_dim=head_dim,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            use_qk_norm=use_qk_norm, qk_norm_eps=qk_norm_eps,
            rope_theta=rope_theta, max_position_embeddings=max_position_embeddings,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.mlp = SwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)


    # Прямой проход через один блок трансформера
    def forward(self, hidden_size: int, norm_eps: float):

        norm1 = self.norm1(self.hidden_size, )