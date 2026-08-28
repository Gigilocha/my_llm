import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Заметка:
(Позиционное кодирование)
Реализация RoPE (Rotary Positional Encoding) - кодирование позиции при помощи вращения векторов эмбеддингов
Cтатья с разбором RoPe: https://llmstudio.ru/blog/rope-rotary-positional-embeddings

Разбиение эмбеддинга на половины дает прирост производительности на 10-15% без потери качества, по сравнению с разбиением на пары.
Подход 1: Рядом идущие
Исходный вектор:  [x0, x1, x2, x3, x4, x5, x6, x7]
Пара 0:          (x0, x1)
Пара 1:          (x2, x3)
Пара 2:          (x4, x5)
Пара 3:          (x6, x7)

Подход 2: Разбиение на половины
Исходный вектор:  [x0, x1, x2, x3 | x4, x5, x6, x7]
Пара 0:          (x0, x4)
Пара 1:          (x1, x5)
Пара 2:          (x2, x6)
Пара 3:          (x3, x7)


(Механизм внимания)
Реализация GQA (Grouped Query Attention) - уменьшение количество голов K и V, головы K и V принимают на себя несколько голов Q
"""


# Вычисление theta от i 
def theta_i(head_dim: int, rope_theta: float) -> torch.Tensor:
    i = torch.arange(0, head_dim, 2).float()
    theta_i = rope_theta ** (-i / head_dim)
    return theta_i

# Вычисление угла вращения RoPe
def rope_matrix(seq_len: int, theta_i: torch.Tensor) -> torch.Tensor:
    m = torch.arange(seq_len).float()
    rotation_angle = torch.outer(m, theta_i)
    return rotation_angle

# Вычисление косинуса и синуса угла вращения
def cos_sin(rotation_angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = rotation_angle.cos()
    sin = rotation_angle.sin()
    return cos, sin

# Добавление вращения к эмбеддингам
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Разбиваем на две половины
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim] # [batch, seq_len, half_dim]
    x2 = x[..., half_dim:] # [batch, seq_len, half_dim]

    # Добавляем размерности для broadcasting
    cos = cos[None, None, :, :] # [1, 1, seq_len, half_dim]
    sin = sin[None, None, :, :] # [1, 1, seq_len, half_dim]

    # Вращение 
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Собираем обратно
    out = torch.cat([rotated_x1, rotated_x2], dim=-1)  # [batch, seq_len, head_dim]
    return out


# Класс мехинизма внимания
class Attention(nn.Module):
    def __init__(self, hidden_size:int, head_dim: int, num_heads: int, num_kv_heads: int, use_qk_norm: bool, qk_norm_eps: float, rope_theta: float, max_position_embeddings: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # Линейные слои
        self.q_layer = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_layer = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_layer = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.output_layer = nn.Linear(self.num_heads * self.head_dim , self.hidden_size, bias=False)

        # Нормализация
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # RoPE
        thetas = theta_i(head_dim, rope_theta)
        angles = rope_matrix(max_position_embeddings, thetas)
        cos, sin = cos_sin(angles)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)


    # Прямой проход
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape

        # Проход через QKV
        q = self.q_layer(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_layer(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_layer(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Нормализация
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Срез до нужной длины
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Востановление голов внимания (нужно увеличить головы k и v до q (4 -> 12))
        repeats = self.num_heads // self.num_kv_heads
        k = torch.repeat_interleave(k, repeats, dim=1)
        v = torch.repeat_interleave(v, repeats, dim=1)

        # SDPA
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Преобразование в изначальную форму
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        
        # Выходной слой
        out = self.output_layer(attn_output)

        return out