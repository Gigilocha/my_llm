# tests/test_lm_eval_adapter.py
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from lm_eval.api.instance import Instance
from src.model.transformer import Transformer
from src.engine.lm_eval_adapter import CustomLMAdapter
from src.tokenizer.tokenizer import encode as tokenizer_encode
from src.engine.generate import generate as naive_generate


def _make_test_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    specials = ["<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"]
    trainer = BpeTrainer(vocab_size=150, special_tokens=specials)
    tok.train_from_iterator(
        ["hello world this is a test of the system", "the cat sat on the mat"] * 20, trainer=trainer,
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|bos|>", eos_token="<|eos|>", pad_token="<|pad|>")


def _make_tiny_model(tokenizer, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(
        vocab_size=tokenizer.vocab_size + 5, num_layer=2, hidden_size=16, head_dim=4,
        num_heads=4, num_kv_heads=2, use_qk_norm=True, qk_norm_eps=1e-6,
        rope_theta=10000.0, max_position_embeddings=64,
        intermediate_size=32, norm_eps=1e-6,
    ).eval()


# loglikelihood() должен давать ТУ ЖЕ математику, что независимый ручной
# forward-проход — критично, потому что бенчмарки типа BLiMP/PIQA/COPA целиком
# полагаются на корректность этого числа, ошибка тут даёт неверный скор молча
def test_loglikelihood_matches_manual_computation():
    tokenizer = _make_test_tokenizer()
    model = _make_tiny_model(tokenizer)
    adapter = CustomLMAdapter(model, tokenizer, device="cpu", max_position_embeddings=64)

    context, continuation = "hello world", " this is a test"
    request = Instance(request_type="loglikelihood", doc={}, arguments=(context, continuation), idx=0)
    [(logprob_adapter, is_greedy_adapter)] = adapter.loglikelihood([request])

    context_ids = tokenizer_encode(tokenizer, context, add_special_tokens=False)
    cont_ids = tokenizer_encode(tokenizer, continuation, add_special_tokens=False)
    full_ids = [tokenizer.bos_token_id] + context_ids + cont_ids

    with torch.no_grad():
        logits = model(torch.tensor([full_ids]))

    manual_logprob = 0.0
    manual_greedy = True
    cont_start = len(full_ids) - len(cont_ids)
    for i, tok_id in enumerate(cont_ids):
        pos = cont_start - 1 + i
        log_probs = F.log_softmax(logits[0, pos, :], dim=-1)
        manual_logprob += log_probs[tok_id].item()
        if log_probs.argmax().item() != tok_id:
            manual_greedy = False

    assert abs(logprob_adapter - manual_logprob) < 1e-4
    assert is_greedy_adapter == manual_greedy


def test_loglikelihood_handles_empty_context():
    tokenizer = _make_test_tokenizer()
    model = _make_tiny_model(tokenizer)
    adapter = CustomLMAdapter(model, tokenizer, device="cpu", max_position_embeddings=64)

    request = Instance(request_type="loglikelihood", doc={}, arguments=("", "hello"), idx=0)
    [(logprob, _)] = adapter.loglikelihood([request])
    assert logprob == logprob  # не NaN


# is_greedy должен быть True РОВНО тогда, когда каждый токен continuation
# совпадает с argmax модели на своей позиции. Проверяем это как инвариант на
# нескольких парах, а не через "сгенерируй greedy и скорь его обратно":
# round-trip decode->encode у BPE не тождественный (спецтокены выбрасываются
# при decode, пробелы склеиваются), поэтому такой тест проверял бы устойчивость
# токенизатора, а не логику адаптера
def test_is_greedy_matches_token_level_argmax_check():
    tokenizer = _make_test_tokenizer()
    model = _make_tiny_model(tokenizer, seed=7)
    adapter = CustomLMAdapter(model, tokenizer, device="cpu", max_position_embeddings=64)

    pairs = [
        ("the cat sat", " on the mat"),
        ("hello world", " this is a test"),
        ("the cat", " hello"),
    ]

    for context, continuation in pairs:
        request = Instance(request_type="loglikelihood", doc={}, arguments=(context, continuation), idx=0)
        [(_, is_greedy_adapter)] = adapter.loglikelihood([request])

        context_ids = tokenizer_encode(tokenizer, context, add_special_tokens=False)
        cont_ids = tokenizer_encode(tokenizer, continuation, add_special_tokens=False)
        full_ids = [tokenizer.bos_token_id] + context_ids + cont_ids

        with torch.no_grad():
            logits = model(torch.tensor([full_ids]))

        cont_start = len(full_ids) - len(cont_ids)
        expected_greedy = all(
            logits[0, cont_start - 1 + i, :].argmax().item() == tok_id
            for i, tok_id in enumerate(cont_ids)
        )
        assert is_greedy_adapter == expected_greedy, f"расхождение на паре {context!r}/{continuation!r}"


def test_loglikelihood_rolling_does_not_crash_and_is_finite():
    tokenizer = _make_test_tokenizer()
    model = _make_tiny_model(tokenizer)
    adapter = CustomLMAdapter(model, tokenizer, device="cpu", max_position_embeddings=64)

    request = Instance(request_type="loglikelihood_rolling", doc={}, arguments=("hello world this is a test",), idx=0)
    [logprob] = adapter.loglikelihood_rolling([request])
    assert logprob == logprob
    assert logprob < 0  # логарифм вероятности всегда <= 0
