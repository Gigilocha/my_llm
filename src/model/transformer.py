import torch
import torch.nn as nn

from src.model.mlp import SwiGLU
from src.model.attention import Attention, build_causal_padding_mask
from src.model.transformer_block import TransformerBlock


# Класс всего трансформера
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, num_layer: int,
                hidden_size: int, head_dim: int,
                num_heads: int, num_kv_heads: int,
                use_qk_norm: bool, qk_norm_eps: float,
                rope_theta: float,
                max_position_embeddings: int,
                intermediate_size: int,
                norm_eps: float,
                tie_word_embeddings: bool = True):
        super().__init__()

        # Эмбеддинг токенов
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Блоки трансформера (было "trasformer_block" — опечатка; к ней уже
        # приходилось обращаться с пояснением "ИСПРАВЛЕНО" в kv_cache.py,
        # здесь фиксится у источника, а не обходится сбоку)
        self.transformer_block = nn.ModuleList([
            TransformerBlock(hidden_size=hidden_size, head_dim=head_dim,
                            num_heads=num_heads, num_kv_heads=num_kv_heads,
                            use_qk_norm=use_qk_norm, qk_norm_eps=qk_norm_eps,
                            rope_theta=rope_theta, max_position_embeddings=max_position_embeddings,
                            intermediate_size=intermediate_size, norm_eps=norm_eps)
            for _ in range(num_layer)
        ])

        # Финальная нормализация
        self.final_norm = nn.RMSNorm(hidden_size, eps=norm_eps)

        # Финальный линейный слой. bias=False — консистентно с остальными
        # линейными слоями модели (Attention/MLP тоже без bias)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017): lm_head использует ТЕ ЖЕ веса,
        # что и входной эмбеддинг, а не свои собственные. nn.Linear.weight
        # имеет форму [out_features, in_features] = [vocab_size, hidden_size] —
        # та же форма, что у nn.Embedding.weight, так что это общий тензор,
        # не копия (градиент от lm_head тоже течёт в эмбеддинг). Экономит
        # vocab_size*hidden_size параметров без потери качества — стандартная
        # практика для моделей такого масштаба (GPT-2 и меньше)
        if tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    # Функция прямого прохода.
    # attention_mask: [batch, seq_len], 1=реальный токен, 0=паддинг. None (по
    # умолчанию) -> обычный причинный forward, как было всегда (pretrain).
    # Задан -> строим причинность+padding маску один раз, передаём во все блоки
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_mask = build_causal_padding_mask(attention_mask) if attention_mask is not None else None

        # Эмбеддинг
        x = self.embedding(x)

        # Блоки трансформера
        for block in self.transformer_block:
            x = block(x, attn_mask=attn_mask)

        # Нормализация
        x = self.final_norm(x)

        # Линейный слой
        x = self.lm_head(x)

        return x