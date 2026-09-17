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
        repetition_penalty: float = 1.3,
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

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
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: bool = True,
    ) -> str:

        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_k = top_k if top_k is not None else self.top_k
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty

        # ИСПРАВЛЕНО: сырой tokenizer.encode() не добавляет BOS (post_processor
        # тут только про byte-level смещения, не про спецтокены) — модель на
        # обучении ВСЕГДА видела BOS на позиции 0, добавляем его здесь явно,
        # как и в src/engine/generate.py
        prompt_ids = tokenizer_encode(self.tokenizer, prompt, add_special_tokens=False)
        generated_ids = [self.tokenizer.bos_token_id] + prompt_ids
        prompt_len = len(generated_ids)  # для return_full_text=False
        input_ids = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
        pos = input_ids.shape[1]

        # Модель не поддерживает позиции дальше max_position_embeddings — RoPE
        # буферы (rope_cos/rope_sin) посчитаны только до этой длины, а KV-кеш,
        # в отличие от src/engine/generate.py, не подрезает окно контекста.
        # Без этой проверки выход за пределы дал бы пустой срез rope_cos и
        # непонятную ошибку вида "shape '[1, 1, N]' is invalid for input of size 0"
        # вместо явного сообщения о причине
        max_position_embeddings = self.model.trasformer_block[0].attention.rope_cos.shape[0]
        if pos > max_position_embeddings:
            raise ValueError(
                f"Промпт ({pos} токенов) длиннее max_position_embeddings модели "
                f"({max_position_embeddings}) — сократи промпт"
            )

        self.reset_cache()

        # Prefill: прогоняем весь промпт (+BOS) целиком
        logits = self._forward_with_cache(input_ids, start_pos=0)

        for _ in range(max_new_tokens):
            # Достигли предела контекста модели — останавливаемся, отдаём то,
            # что уже сгенерировано, вместо падения на пустом RoPE-срезе
            if pos >= max_position_embeddings:
                break

            next_logits = logits[:, -1, :].clone()

            # Repetition penalty — тот же принцип, что в src/engine/generate.py:
            # штрафуем токены, которые уже встречались в generated_ids (промпт +
            # уже сгенерированное), чтобы не проваливаться в буквальные повторы
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            next_logits = next_logits / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                top_k_eff = min(top_k, next_logits.size(-1))
                indices_to_remove = next_logits < torch.topk(next_logits, top_k_eff)[0][..., -1, None]
                next_logits[indices_to_remove] = -float("inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_mask] = -float("inf")
                next_logits = torch.full_like(next_logits, -float("inf"))
                next_logits.scatter_(1, sorted_indices, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            if next_id.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_id.item())

            # Decode: прогоняем только новый токен с кешем
            logits = self._forward_with_cache(next_id, start_pos=pos)
            pos += 1

        return tokenizer_decode(self.tokenizer, generated_ids if return_full_text else generated_ids[prompt_len:])

    # Память, занятая KV-кешем (в MB).
    def cache_memory_mb(self) -> float:
        total = 0
        for cache in self.caches:
            if cache.k is not None:
                total += cache.k.numel() * cache.k.element_size()
                total += cache.v.numel() * cache.v.element_size()
        return total / (1024 ** 2)
