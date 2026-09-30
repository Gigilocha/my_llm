from src.model.transformer import Transformer


# Общая сборка модели из конфига. Раньше эта функция была дословно
# продублирована в шести местах (base_train.py, base_eval.py, sft_train.py,
# sft_eval.py, rlft_train.py, chat.py) без единого отличия между копиями —
# в отличие от кеширования данных, тут даже не было косметических различий,
# поэтому объединение — просто перенос, а не обобщение
def build_model(config) -> Transformer:
    return Transformer(
        vocab_size=config.model.model.vocab_size,
        num_layer=config.model.model.num_layers,
        hidden_size=config.model.model.hidden_size,
        head_dim=config.model.attention.head_dim,
        num_heads=config.model.attention.num_heads,
        num_kv_heads=config.model.attention.num_kv_heads,
        use_qk_norm=config.model.attention.use_qk_norm,
        qk_norm_eps=config.model.attention.qk_norm_eps,
        rope_theta=config.model.attention.rope_theta,
        max_position_embeddings=config.model.model.max_position_embeddings,
        intermediate_size=config.model.mlp.intermediate_size,
        norm_eps=config.model.model.norm_eps,
    )

