from typing import Callable

from src.engine.generate import generate as naive_generate
from src.engine.kv_cache import GenerationEngine


"""
Единая точка выбора движка генерации (config.engine.use_kv_cache), чтобы
скрипты (base_eval.py, sft_eval.py) не дублировали ветвление "какой движок
использовать" и не расходились в дефолтах — оба всегда читают engine_config.yaml
через один и тот же путь.
"""


def make_generate_fn(model, tokenizer, device: str, max_position_embeddings: int, engine_cfg) -> Callable[..., str]:
    if engine_cfg.use_kv_cache:
        engine = GenerationEngine(
            model, tokenizer, device=device,
            max_new_tokens=engine_cfg.max_new_tokens,
            temperature=engine_cfg.temperature,
            top_k=engine_cfg.top_k,
            repetition_penalty=engine_cfg.repetition_penalty,
        )

        def _generate(prompt: str, **overrides) -> str:
            return engine.generate(prompt, top_p=engine_cfg.top_p, **overrides)

        return _generate

    def _generate(prompt: str, **overrides) -> str:
        kwargs = dict(
            max_position_embeddings=max_position_embeddings,
            max_new_tokens=engine_cfg.max_new_tokens,
            temperature=engine_cfg.temperature,
            top_k=engine_cfg.top_k,
            top_p=engine_cfg.top_p,
            repetition_penalty=engine_cfg.repetition_penalty,
        )
        kwargs.update(overrides)
        return naive_generate(model, tokenizer, prompt, device, **kwargs)

    return _generate
