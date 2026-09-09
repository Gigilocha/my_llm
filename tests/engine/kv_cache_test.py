# tests/test_kv_cache.py
import torch

from src.model.transformer import Transformer
from src.engine.kv_cache import GenerationEngine, KVCache


def _make_tiny_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(
        vocab_size=64, num_layer=2, hidden_size=16, head_dim=4,
        num_heads=4, num_kv_heads=2, use_qk_norm=True, qk_norm_eps=1e-6,
        rope_theta=10000.0, max_position_embeddings=32,
        intermediate_size=32, norm_eps=1e-6,
    ).eval()


# Главная проверка: кешированный forward должен давать ТЕ ЖЕ логиты, что и
# обычный forward всей последовательности целиком — иначе KV-кеш меняет
# поведение модели, а не просто ускоряет вычисление того же самого
def test_kv_cache_matches_uncached_forward():
    model = _make_tiny_model()

    torch.manual_seed(1)
    full_seq = torch.randint(0, 64, (1, 12))

    with torch.no_grad():
        logits_full = model(full_seq)

    engine = GenerationEngine.__new__(GenerationEngine)  # без tokenizer, нужен только _forward_with_cache
    engine.model = model
    engine.device = "cpu"
    engine.caches = [KVCache() for _ in range(len(model.trasformer_block))]

    prefill_len = 5
    with torch.no_grad():
        logits_cached = engine._forward_with_cache(full_seq[:, :prefill_len], start_pos=0)
    outputs = [logits_cached]
    pos = prefill_len
    with torch.no_grad():
        for i in range(prefill_len, full_seq.shape[1]):
            step_logits = engine._forward_with_cache(full_seq[:, i:i + 1], start_pos=pos)
            outputs.append(step_logits)
            pos += 1

    logits_cached_full = torch.cat(outputs, dim=1)

    assert logits_full.shape == logits_cached_full.shape
    assert (logits_full - logits_cached_full).abs().max().item() < 1e-3
    assert (logits_full.argmax(dim=-1) == logits_cached_full.argmax(dim=-1)).all()


# Кеш должен расти по длине последовательности на каждом слое и корректно
# сбрасываться в исходное (пустое) состояние
def test_kv_cache_append_and_reset():
    cache = KVCache()
    assert cache.k is None

    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)
    cache.append(k1, v1)
    assert cache.k.shape == (1, 2, 3, 4)

    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)
    cache.append(k2, v2)
    assert cache.k.shape == (1, 2, 4, 4)  # 3 + 1 по seq_len (dim=2)

    cache.reset()
    assert cache.k is None
    assert cache.v is None


# Число кешей должно совпадать с реальным количеством слоёв модели —
# это то место, где раньше был баг (model.num_layers не существовал)
def test_kv_cache_count_matches_model_layers():
    model = _make_tiny_model()
    engine = GenerationEngine.__new__(GenerationEngine)
    engine.model = model
    engine.caches = [KVCache() for _ in range(len(model.trasformer_block))]
    assert len(engine.caches) == 2  # num_layer=2 в _make_tiny_model
