import torch
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from src.engine.loglikelihood import sequence_loglikelihood
from src.engine.select import make_generate_fn


"""
Адаптер под интерфейс lm_eval.api.model.LM — позволяет прогонять нашу модель
через lm-evaluation-harness (EleutherAI), не переписывая логику бенчмарков
самим. Наша Transformer — не HF AutoModelForCausalLM, поэтому напрямую
--model hf не заведётся; вместо этого создаём готовый объект и передаём
в lm_eval.simple_evaluate(model=<этот объект>, ...).

loglikelihood() — единственный метод, который реально нужен для бенчмарков
типа BLiMP/PIQA/COPA/PARus (сравнение log-вероятности нескольких вариантов
продолжения, не свободная генерация) — устойчиво работает даже на слабо
обученной модели, в отличие от бенчмарков, требующих связной генерации.

Сам подсчёт log-вероятности — sequence_loglikelihood() в src/engine/
loglikelihood.py, общая с benchmark.py (прямой скоринг ruMMLU-pro/MMLU-Pro
без обвязки harness, см. benchmark.py) — раньше эта логика здесь была
единственной копией, полностью дублировавшейся, будь она нужна где-то ещё.
"""


class CustomLMAdapter(LM):
    def __init__(self, model, tokenizer, device: str, max_position_embeddings: int, engine_cfg=None):
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._device = device
        self.max_position_embeddings = max_position_embeddings
        self.engine_cfg = engine_cfg

    # (context, continuation) -> (logprob суммы токенов continuation, is_greedy)
    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        for request in tqdm(requests, desc="loglikelihood"):
            context, continuation = request.args
            results.append(
                sequence_loglikelihood(self.model, self.tokenizer, context, continuation, self.device, self.max_position_embeddings)
            )
        return results

    # Полная log-вероятность строки целиком (для перплексии) — не используется
    # выбранными бенчмарками (BLiMP/PIQA/COPA/PARus все идут через loglikelihood),
    # но это abstract-метод базового класса, нужна рабочая реализация.
    # sequence_loglikelihood(context="", continuation=text) даёт ровно ту же
    # величину: логиты позиции i предсказывают токен i+1, начиная с BOS —
    # то есть каждый токен текста предсказывается по всем предыдущим
    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        results = []
        for request in tqdm(requests, desc="loglikelihood_rolling"):
            (text,) = request.args
            logprob, _ = sequence_loglikelihood(self.model, self.tokenizer, "", text, self.device, self.max_position_embeddings)
            results.append(logprob)
        return results

    # Свободная генерация до стоп-последовательности — используется малой
    # частью задач в некоторых бенчмарках, не основным набором (BLiMP/PIQA/
    # COPA/PARus его не требуют вообще, они целиком через loglikelihood)
    def generate_until(self, requests: list[Instance]) -> list[str]:
        generate_fn = make_generate_fn(self.model, self.tokenizer, self.device, self.max_position_embeddings, self.engine_cfg)
        results = []
        for request in tqdm(requests, desc="generate_until"):
            context, gen_kwargs = request.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            max_new_tokens = gen_kwargs.get("max_gen_toks", 128) if isinstance(gen_kwargs, dict) else 128

            text = generate_fn(context, max_new_tokens=max_new_tokens, return_full_text=False)
            for stop in until:
                if stop in text:
                    text = text[:text.index(stop)]
            results.append(text)

        return results