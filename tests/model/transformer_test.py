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