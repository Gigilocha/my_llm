from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.model.transformer import Transformer
from src.model.attention import apply_rope
from transformers import PreTrainedTokenizerFast
from src.tokenizer.tokenizer import encode as tokenizer_encode, decode as tokenizer_decode


# KV-кеш для одного слоя.
@dataclass
class KVCache:
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def reset(self):
        self.k = None
        self.v = None


# Инференс с KV-кешем.
# Не трогает тренировочный forward модели.
class GenerationEngine:
    def __init__(
        self,
        model: Transformer,
        tokenizer: PreTrainedTokenizerFast,
        device: str = "cuda",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

        # ИСПРАВЛЕНО: model.num_layers не существует — берём длину уже
        # построенного ModuleList, а не несуществующий сохранённый атрибут
        self.caches = [KVCache() for _ in range(len(model.trasformer_block))]

    def reset_cache(self):
        for cache in self.caches:
            cache.reset()

    # input_ids: (batch, new_len) — только НОВЫЕ токены.
    # start_pos: позиция, с которой начинаются новые токены (для RoPE).
    @torch.no_grad()
    def _forward_with_cache(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        x = self.model.embedding(input_ids)
        batch, new_len, _ = x.shape

        # ИСПРАВЛЕНО: атрибут называется trasformer_block (опечатка в исходном
        # классе Transformer), а не transformer_block
        for layer_idx, block in enumerate(self.model.trasformer_block):
            attn = block.attention
            norm_out = block.norm1(x)

            # QKV проекции
            q = attn.q_layer(norm_out).view(batch, new_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_layer(norm_out).view(batch, new_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = attn.v_layer(norm_out).view(batch, new_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

            # ИСПРАВЛЕНО: attn.use_qk_norm не существует как атрибут — q_norm/k_norm
            # это либо RMSNorm, либо nn.Identity() (решается один раз при
            # конструировании), вызываем БЕЗ условия, как в оригинальном forward()
            q = attn.q_norm(q)
            k = attn.k_norm(k)

            # RoPE
            cos = attn.rope_cos[start_pos:start_pos + new_len]
            sin = attn.rope_sin[start_pos:start_pos + new_len]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

            # Добавляем в кеш
            k_full, v_full = self.caches[layer_idx].append(k, v)

            # GQA: повторяем головы KV до числа Q-голов
            repeats = attn.num_heads // attn.num_kv_heads
            if repeats > 1:
                k_rep = k_full.repeat_interleave(repeats, dim=1)
                v_rep = v_full.repeat_interleave(repeats, dim=1)
            else:
                k_rep, v_rep = k_full, v_full

            # SDPA. is_causal=True только на prefill (new_len>1, нужна маска
            # внутри промпта) — на decode (new_len==1) кеш содержит только
            # позиции <= текущей, маскировать нечего, поэтому is_causal=False
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                is_causal=(new_len > 1)
            )

            out = out.transpose(1, 2).contiguous().view(batch, new_len, attn.num_heads * attn.head_dim)
            x = x + attn.output_layer(out)

            # MLP
            x = x + block.mlp(block.norm2(x))

        x = self.model.final_norm(x)
        logits = self.model.lm_head(x)
        return logits

    # Генерирует текст по промпту.
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:

        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k

        # ИСПРАВЛЕНО: сырой tokenizer.encode() не добавляет BOS (post_processor
        # тут только про byte-level смещения, не про спецтокены) — модель на
        # обучении ВСЕГДА видела BOS на позиции 0, добавляем его здесь явно,
        # как и в src/engine/generate.py
        prompt_ids = tokenizer_encode(self.tokenizer, prompt, add_special_tokens=False)
        generated_ids = [self.tokenizer.bos_token_id] + prompt_ids
        input_ids = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
        pos = input_ids.shape[1]

        self.reset_cache()

        # Prefill: прогоняем весь промпт (+BOS) целиком
        logits = self._forward_with_cache(input_ids, start_pos=0)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                top_k_eff = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, top_k_eff)[0][..., -1, None]
                next_logits[indices_to_remove] = -float("inf")

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            if next_id.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_id.item())

            # Decode: прогоняем только новый токен с кешем
            logits = self._forward_with_cache(next_id, start_pos=pos)
            pos += 1

        return tokenizer_decode(self.tokenizer, generated_ids)

    # Память, занятая KV-кешем (в MB).
    def cache_memory_mb(self) -> float:
        total = 0
        for cache in self.caches:
            if cache.k is not None:
                total += cache.k.numel() * cache.k.element_size()
                total += cache.v.numel() * cache.v.element_size()
        return total / (1024 ** 2)
