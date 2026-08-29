import torch
from src.model.transformer_block import TransformerBlock


# Тест
def test_transformer_block_residual_preserves_input_when_sublayers_are_zero():
    batch, seq_len = 2, 128
    hidden_size, head_dim, num_heads, num_kv_heads = 768, 64, 12, 4
    use_qk_norm, qk_norm_eps = True, 1e-6
    rope_theta, max_position_embeddings = 10000.0, 6144
    intermediate_size, norm_eps = 2048, 1e-6

    block = TransformerBlock(hidden_size, head_dim, num_heads, num_kv_heads,
                              use_qk_norm, qk_norm_eps, rope_theta, max_position_embeddings,
                              intermediate_size, norm_eps)

    # Обнуляем выходные слои attention и mlp — гарантируем, что их вклад = 0
    with torch.no_grad():
        block.attention.output_layer.weight.zero_()
        block.mlp.down.weight.zero_()

    x = torch.randn(batch, seq_len, hidden_size)
    output = block(x)

    assert torch.allclose(output, x, atol=1e-5)