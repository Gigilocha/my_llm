import torch
from src.model.transformer import Transformer


def test_transformer_output_shape():
    batch, seq_len = 2, 128
    vocab_size, num_layer = 65536, 12
    hidden_size, head_dim, num_heads, num_kv_heads = 768, 64, 12, 4
    use_qk_norm, qk_norm_eps = True, 1e-6
    rope_theta, max_position_embeddings = 10000.0, 6144
    intermediate_size, norm_eps = 2048, 1e-6

    model = Transformer(vocab_size, num_layer, hidden_size, head_dim, num_heads, num_kv_heads,
                        use_qk_norm, qk_norm_eps, rope_theta, max_position_embeddings,
                        intermediate_size, norm_eps)

    input_ids = torch.randint(0, vocab_size, (batch, seq_len))  # случайные ID токенов
    output = model(input_ids)

    assert output.shape == (batch, seq_len, vocab_size)

import torch
from src.model.transformer import Transformer


def _make_tiny_transformer(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(
        vocab_size=64, num_layer=2, hidden_size=16, head_dim=4,
        num_heads=4, num_kv_heads=2, use_qk_norm=True, qk_norm_eps=1e-6,
        rope_theta=10000.0, max_position_embeddings=32,
        intermediate_size=32, norm_eps=1e-6,
    ).eval()


# attention_mask из одних единиц (нет паддинга) должен давать РОВНО тот же
# результат, что и attention_mask=None (обычный pretrain-путь) — иначе SFT
# незаметно меняет поведение модели даже там, где паддинга нет вообще
def test_attention_mask_all_ones_matches_none():
    model = _make_tiny_transformer()
    torch.manual_seed(1)
    x = torch.randint(0, 64, (2, 10))

    with torch.no_grad():
        logits_no_mask = model(x)
        logits_all_ones = model(x, attention_mask=torch.ones(2, 10, dtype=torch.long))

    assert torch.allclose(logits_no_mask, logits_all_ones, atol=1e-5)


# Паддинг в конце последовательности не должен влиять на логиты РЕАЛЬНЫХ
# (непаддинговых) токенов — это и есть смысл маски
def test_padding_does_not_affect_real_token_logits():
    model = _make_tiny_transformer()
    torch.manual_seed(2)
    real_len = 7
    pad_len = 3
    x_real = torch.randint(0, 64, (1, real_len))
    x_padded = torch.cat([x_real, torch.zeros(1, pad_len, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, real_len, dtype=torch.long), torch.zeros(1, pad_len, dtype=torch.long)], dim=1)

    with torch.no_grad():
        logits_real_only = model(x_real)
        logits_padded = model(x_padded, attention_mask=mask)

    # Сравниваем логиты только на позициях реальных токенов (первые real_len)
    assert torch.allclose(logits_real_only, logits_padded[:, :real_len, :], atol=1e-5)
