import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

from src.tokenizer.tokenizer import encode as tokenizer_encode
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

            context_ids = tokenizer_encode(self.tokenizer, context, add_special_tokens=False) if context else []
            continuation_ids = tokenizer_encode(self.tokenizer, continuation, add_special_tokens=False)

            full_ids = [self.tokenizer.bos_token_id] + context_ids + continuation_ids
            cont_len = len(continuation_ids)

            # Обрезаем СЛЕВА (старый контекст), если не влезает — continuation
            # должен остаться в контексте целиком, иначе нечего скорить
            if len(full_ids) > self.max_position_embeddings:
                full_ids = full_ids[-self.max_position_embeddings:]

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            logits = self.model(input_ids)  # [1, seq_len, vocab]

            # Логиты на позиции i предсказывают токен i+1 — continuation начинается
            # с позиции (len(full_ids) - cont_len), значит предсказывающие его
            # логиты лежат на один индекс раньше
            cont_start = len(full_ids) - cont_len
            pred_logits = logits[0, cont_start - 1: -1, :]  # [cont_len, vocab]
            target_ids = torch.tensor(full_ids[cont_start:], dtype=torch.long, device=self.device)

            log_probs = F.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            total_logprob = token_log_probs.sum().item()

            greedy_ids = pred_logits.argmax(dim=-1)
            is_greedy = bool((greedy_ids == target_ids).all().item())

            results.append((total_logprob, is_greedy))

        return results

    # Полная log-вероятность строки целиком (для перплексии) — не используется
    # выбранными бенчмарками (BLiMP/PIQA/COPA/PARus все идут через loglikelihood),
    # но это abstract-метод базового класса, нужна рабочая реализация
    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        results = []
        for request in tqdm(requests, desc="loglikelihood_rolling"):
            (text,) = request.args
            ids = [self.tokenizer.bos_token_id] + tokenizer_encode(self.tokenizer, text, add_special_tokens=False)
            ids = ids[-self.max_position_embeddings:]

            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model(input_ids)

            pred_logits = logits[0, :-1, :]
            target_ids = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
            log_probs = F.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            results.append(token_log_probs.sum().item())

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
