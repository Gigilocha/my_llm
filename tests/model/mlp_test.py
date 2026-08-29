import torch
from src.model.mlp import SwiGLU


# Тест формы выхода, что форма входа совпадает с формой входа
def test_swiglu_output_shape():
    batch, seq_len, hidden_size, intermediate_size = 2, 10, 768, 2048

    x = torch.randn(batch, seq_len, hidden_size)
    mlp = SwiGLU(hidden_size, intermediate_size)

    output = mlp(x)

    assert output.shape == x.shape


# Тест выхода не NaN/In
def test_swiglu_output_has_no_nan_or_inf() -> bool:
    batch, seq_len, hidden_size, intermediate_size = 2, 10, 768, 2048

    x = torch.randn(batch, seq_len, hidden_size)
    mlp = SwiGLU(hidden_size, intermediate_size)
    
    output = mlp(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    
    