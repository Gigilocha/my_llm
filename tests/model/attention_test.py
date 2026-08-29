import torch
from src.model.attention import Attention 


# Проверка форм выхода и входа
def test_attention_output_shape():
    batch, seq_len = 2, 128
    hidden_size, head_dim, num_heads, num_kv_heads, use_qk_norm, qk_eps, rope_theta, max_position_embeddings = 768, 64, 12, 4, True, 0.000001, 10000.0, 6144

    x = torch.randn(batch, seq_len, hidden_size)
    attention = Attention(hidden_size, head_dim, num_heads, num_kv_heads, use_qk_norm, qk_eps, rope_theta, max_position_embeddings)

    output = attention(x)

    assert output.shape == x.shape


# Тест на причинно-следственные связи
def test_attention_is_causal():
    batch, seq_len = 2, 128
    hidden_size, head_dim, num_heads, num_kv_heads, use_qk_norm, qk_eps, rope_theta, max_position_embeddings = 768, 64, 12, 4, True, 0.000001, 10000.0, 6144

    attention = Attention(hidden_size, head_dim, num_heads, num_kv_heads, use_qk_norm, qk_eps, rope_theta, max_position_embeddings)

    attention.eval()  # отключаем dropout/etc, если есть — для детерминированности

    x1 = torch.randn(batch, seq_len, hidden_size)
    x2 = x1.clone()
    x2[:, 100, :] = torch.randn(batch, hidden_size)  # меняем ТОЛЬКО позднюю позицию 100

    with torch.no_grad():
        out1 = attention(x1)
        out2 = attention(x2)

    # ранняя позиция (5) не должна измениться — она не видит будущее (позицию 100)
    assert torch.allclose(out1[:, 5, :], out2[:, 5, :], atol=1e-5)

    # поздняя позиция (100 и дальше) ДОЛЖНА измениться — видит саму себя/то, что изменилось
    assert not torch.allclose(out1[:, 100, :], out2[:, 100, :], atol=1e-5)


# Тест 
def test_attention_works_without_qk_norm():
    batch, seq_len = 2, 128
    hidden_size, head_dim, num_heads, num_kv_heads = 768, 64, 12, 4
    rope_theta, max_position_embeddings = 10000.0, 6144

    attention = Attention(hidden_size, head_dim, num_heads, num_kv_heads,
                          use_qk_norm=False, qk_norm_eps=1e-6,
                          rope_theta=rope_theta, max_position_embeddings=max_position_embeddings)

    x = torch.randn(batch, seq_len, hidden_size)
    output = attention(x)

    assert output.shape == x.shape