# tests/test_dpo.py
import math

import pytest
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from src.common.config import DPOSource
from src.data.dpo_dataset import _extract_dpo_example, _peek_and_validate_fields
from src.data.dpo_format import tokenize_dpo_example
from src.data.sft_format import IGNORE_INDEX
from src.data.sft_dataloader import collate_sft_batch
from src.training.dpo_train_step import sequence_logprob, dpo_train_step
from src.model.transformer import Transformer


SPECIAL_TOKENS = {
    "bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>",
    "user_start": "<|user_start|>", "user_end": "<|user_end|>",
    "assistant_start": "<|assistant_start|>", "assistant_end": "<|assistant_end|>",
}


def _make_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    vocab = ["<|unk|>"] + list(SPECIAL_TOKENS.values())
    tok.add_special_tokens(vocab)
    tok.train_from_iterator(["hello world good bad answer question"] * 20,
                            trainer=BpeTrainer(vocab_size=200, special_tokens=vocab))
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=SPECIAL_TOKENS["bos"], eos_token=SPECIAL_TOKENS["eos"],
        pad_token=SPECIAL_TOKENS["pad"],
        additional_special_tokens=[v for k, v in SPECIAL_TOKENS.items() if k not in ("bos", "eos", "pad")],
    )


def _make_model(tokenizer, seed=0):
    torch.manual_seed(seed)
    return Transformer(
        vocab_size=tokenizer.vocab_size + 10, num_layer=2, hidden_size=16, head_dim=4,
        num_heads=4, num_kv_heads=2, use_qk_norm=True, qk_norm_eps=1e-6,
        rope_theta=10000.0, max_position_embeddings=128, intermediate_size=32, norm_eps=1e-6,
    )


# --- Извлечение данных ---

def test_extract_from_plain_string_fields():
    source = DPOSource(dataset_name="x")
    doc = {"prompt": "question", "chosen": "good answer", "rejected": "bad answer"}
    ex = _extract_dpo_example(doc, source)
    assert ex["messages"] == [{"role": "user", "content": "question"}]
    assert ex["chosen"] == "good answer"
    assert ex["rejected"] == "bad answer"


def test_extract_from_messages_format_takes_last_assistant():
    source = DPOSource(dataset_name="x", as_messages=True)
    doc = {
        "prompt": [{"role": "user", "content": "question"}],
        "chosen": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "good"}],
        "rejected": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "bad"}],
    }
    ex = _extract_dpo_example(doc, source)
    assert ex["chosen"] == "good"
    assert ex["rejected"] == "bad"


def test_extract_drops_identical_and_empty_pairs():
    source = DPOSource(dataset_name="x")
    # Одинаковые ответы — нулевой градиент DPO, пара бесполезна
    assert _extract_dpo_example({"prompt": "q", "chosen": "same", "rejected": "same"}, source) is None
    assert _extract_dpo_example({"prompt": "q", "chosen": "", "rejected": "bad"}, source) is None
    assert _extract_dpo_example({"prompt": "", "chosen": "a", "rejected": "b"}, source) is None


def test_prompt_messages_strip_trailing_assistant_turn():
    source = DPOSource(dataset_name="x", as_messages=True)
    doc = {
        "prompt": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "leaked answer"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }
    ex = _extract_dpo_example(doc, source)
    # Хвостовой ответ из промпта убран — иначе он утёк бы в обе ветки
    assert all(m["role"] != "assistant" for m in ex["messages"])


def test_peek_validate_reports_missing_fields():
    source = DPOSource(dataset_name="x", chosen_field="preferred")
    with pytest.raises(ValueError, match="preferred"):
        _peek_and_validate_fields(iter([{"prompt": "q", "chosen": "a", "rejected": "b"}]), source)


# --- Токенизация пары ---

def test_tokenize_dpo_example_shares_prompt_and_masks_it():
    tokenizer = _make_tokenizer()
    example = {"messages": [{"role": "user", "content": "question"}], "chosen": "good", "rejected": "bad"}
    result = tokenize_dpo_example(tokenizer, example, SPECIAL_TOKENS, max_len=64)
    assert result is not None
    (c_ids, c_labels), (r_ids, r_labels) = result

    # Общий префикс промпта должен совпадать в обеих ветках
    prompt_len = sum(1 for l in c_labels if l == IGNORE_INDEX)
    assert c_ids[:prompt_len] == r_ids[:prompt_len]
    # И быть замаскирован в обеих
    assert all(l == IGNORE_INDEX for l in c_labels[:prompt_len])
    assert all(l == IGNORE_INDEX for l in r_labels[:prompt_len])
    # Ответы видны
    assert any(l != IGNORE_INDEX for l in c_labels)
    assert any(l != IGNORE_INDEX for l in r_labels)


# --- Математика: sequence_logprob ---

def test_sequence_logprob_matches_manual_sum_over_response_only():
    tokenizer = _make_tokenizer()
    model = _make_model(tokenizer).eval()

    input_ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7, 8, 9]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        got = sequence_logprob(model, input_ids, labels, attention_mask)

        logits = model(input_ids[:, :-1], attention_mask=attention_mask[:, :-1])
        log_probs = F.log_softmax(logits.float(), dim=-1)
        targets = labels[:, 1:]
        manual = sum(
            log_probs[0, i, targets[0, i]].item()
            for i in range(targets.shape[1]) if targets[0, i] != IGNORE_INDEX
        )

    assert abs(got.item() - manual) < 1e-4


def test_sequence_logprob_ignores_padding_positions():
    tokenizer = _make_tokenizer()
    model = _make_model(tokenizer).eval()

    real_ids = torch.tensor([[1, 5, 6, 7]])
    real_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 6, 7]])
    real_mask = torch.ones_like(real_ids)

    padded_ids = torch.tensor([[1, 5, 6, 7, 0, 0]])
    padded_labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 6, 7, IGNORE_INDEX, IGNORE_INDEX]])
    padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])

    with torch.no_grad():
        a = sequence_logprob(model, real_ids, real_labels, real_mask)
        b = sequence_logprob(model, padded_ids, padded_labels, padded_mask)

    assert abs(a.item() - b.item()) < 1e-4


# --- Математика: DPO loss ---

def test_dpo_loss_equals_log2_when_policy_equals_reference():
    """Если политика идентична reference, dpo_logits=0 и loss = -log(sigmoid(0)) = log(2)."""
    tokenizer = _make_tokenizer()
    policy = _make_model(tokenizer, seed=3)
    ref = _make_model(tokenizer, seed=3)  # ТЕ ЖЕ веса
    ref.eval()

    example = {"messages": [{"role": "user", "content": "question"}], "chosen": "good", "rejected": "bad"}
    (c, r) = tokenize_dpo_example(tokenizer, example, SPECIAL_TOKENS, max_len=64)
    c_ids, c_labels, c_mask = collate_sft_batch([c], tokenizer.pad_token_id)
    r_ids, r_labels, r_mask = collate_sft_batch([r], tokenizer.pad_token_id)

    loss, metrics = dpo_train_step(policy, ref, c_ids, c_labels, c_mask, r_ids, r_labels, r_mask, beta=0.1)

    assert abs(loss - math.log(2)) < 1e-4, f"ожидался log(2)={math.log(2):.6f}, получен {loss:.6f}"
    assert abs(metrics["reward_margin"]) < 1e-5


def test_dpo_gradient_pushes_chosen_up_and_rejected_down():
    """После шага DPO logprob(chosen) должен вырасти относительно logprob(rejected)."""
    tokenizer = _make_tokenizer()
    policy = _make_model(tokenizer, seed=3)
    ref = _make_model(tokenizer, seed=3)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    example = {"messages": [{"role": "user", "content": "question"}], "chosen": "good", "rejected": "bad"}
    (c, r) = tokenize_dpo_example(tokenizer, example, SPECIAL_TOKENS, max_len=64)
    c_ids, c_labels, c_mask = collate_sft_batch([c], tokenizer.pad_token_id)
    r_ids, r_labels, r_mask = collate_sft_batch([r], tokenizer.pad_token_id)

    with torch.no_grad():
        margin_before = (sequence_logprob(policy, c_ids, c_labels, c_mask)
                         - sequence_logprob(policy, r_ids, r_labels, r_mask)).item()

    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
    for _ in range(5):
        optimizer.zero_grad()
        dpo_train_step(policy, ref, c_ids, c_labels, c_mask, r_ids, r_labels, r_mask, beta=0.1)
        optimizer.step()

    with torch.no_grad():
        margin_after = (sequence_logprob(policy, c_ids, c_labels, c_mask)
                        - sequence_logprob(policy, r_ids, r_labels, r_mask)).item()

    assert margin_after > margin_before, f"margin не вырос: было {margin_before:.4f}, стало {margin_after:.4f}"


def test_reference_model_stays_frozen():
    tokenizer = _make_tokenizer()
    policy = _make_model(tokenizer, seed=3)
    ref = _make_model(tokenizer, seed=3)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    ref_snapshot = [p.detach().clone() for p in ref.parameters()]

    example = {"messages": [{"role": "user", "content": "question"}], "chosen": "good", "rejected": "bad"}
    (c, r) = tokenize_dpo_example(tokenizer, example, SPECIAL_TOKENS, max_len=64)
    c_ids, c_labels, c_mask = collate_sft_batch([c], tokenizer.pad_token_id)
    r_ids, r_labels, r_mask = collate_sft_batch([r], tokenizer.pad_token_id)

    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
    optimizer.zero_grad()
    dpo_train_step(policy, ref, c_ids, c_labels, c_mask, r_ids, r_labels, r_mask, beta=0.1)
    optimizer.step()

    for before, after in zip(ref_snapshot, ref.parameters()):
        assert torch.equal(before, after), "reference-модель изменилась — регуляризация сломана!"
        assert after.grad is None
