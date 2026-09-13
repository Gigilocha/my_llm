# tests/test_sft_format.py
import pytest
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from src.data.sft_format import tokenize_sft_example, format_prompt_for_generation, IGNORE_INDEX
from src.data.sft_dataloader import collate_sft_batch
from src.training.sft_train_step import sft_train_step
from src.model.transformer import Transformer
from src.common.config import SFTSource


SPECIAL_TOKENS = {
    "bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>",
    "user_start": "<|user_start|>", "user_end": "<|user_end|>",
    "assistant_start": "<|assistant_start|>", "assistant_end": "<|assistant_end|>",
}


def _make_test_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer_vocab = ["<|unk|>"] + list(SPECIAL_TOKENS.values())
    tok.add_special_tokens(trainer_vocab)
    trainer = BpeTrainer(vocab_size=200, special_tokens=trainer_vocab)
    tok.train_from_iterator(
        ["hello world this is a test", "напиши функцию на питоне", "def foo(): return 1"] * 20,
        trainer=trainer,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=SPECIAL_TOKENS["bos"], eos_token=SPECIAL_TOKENS["eos"], pad_token=SPECIAL_TOKENS["pad"],
        additional_special_tokens=[v for k, v in SPECIAL_TOKENS.items() if k not in ("bos", "eos", "pad")],
    )


# --- Конфиг источника: должен требовать РОВНО один формат ---

def test_sft_source_requires_exactly_one_format():
    # Ни один формат не задан -> ошибка
    with pytest.raises(ValueError):
        SFTSource(dataset_name="x")

    # Оба формата разом -> тоже ошибка
    with pytest.raises(ValueError):
        SFTSource(dataset_name="x", messages_field="messages", instruction_field="instruction", output_field="output")

    # Ровно один -> ок
    SFTSource(dataset_name="x", messages_field="messages")
    SFTSource(dataset_name="y", instruction_field="instruction", output_field="output")


# --- Токенизация: single-turn ---

def test_tokenize_single_turn_masks_prompt_correctly():
    tokenizer = _make_test_tokenizer()
    example = {"messages": [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "this is a test"},
    ]}

    result = tokenize_sft_example(tokenizer, example, SPECIAL_TOKENS, max_len=64)
    assert result is not None
    input_ids, labels = result

    assert len(input_ids) == len(labels)
    assert labels[0] == IGNORE_INDEX  # BOS всегда замаскирован
    assert any(l != IGNORE_INDEX for l in labels)  # где-то есть видимый ответ
    for i, l in enumerate(labels):
        if l != IGNORE_INDEX:
            assert l == input_ids[i]


# --- Токенизация: multi-turn (два user/assistant хода подряд) ---

def test_tokenize_multi_turn_masks_each_turn_correctly():
    tokenizer = _make_test_tokenizer()
    example = {"messages": [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "this is a test"},
        {"role": "user", "content": "напиши функцию"},
        {"role": "assistant", "content": "def foo(): return 1"},
    ]}

    result = tokenize_sft_example(tokenizer, example, SPECIAL_TOKENS, max_len=128)
    assert result is not None
    input_ids, labels = result

    # Должно быть ДВА непрерывных видимых блока (два ответа ассистента),
    # разделённых замаскированным участком (второй user-ход)
    visible = [l != IGNORE_INDEX for l in labels]
    # Находим границы блоков видимости
    blocks = []
    in_block = False
    for v in visible:
        if v and not in_block:
            blocks.append(1)
            in_block = True
        elif not v:
            in_block = False
    assert len(blocks) == 2, f"Ожидалось 2 видимых блока (два ответа), получено {len(blocks)}"


# --- Системное сообщение маскируется, не участвует в лоссе ---

def test_system_message_is_masked():
    tokenizer = _make_test_tokenizer()
    example = {"messages": [
        {"role": "system", "content": "hello world test"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "this is a test"},
    ]}
    result = tokenize_sft_example(tokenizer, example, SPECIAL_TOKENS, max_len=64)
    assert result is not None
    _, labels = result
    # Первые несколько токенов (system) должны быть замаскированы
    assert labels[0] == IGNORE_INDEX
    assert labels[1] == IGNORE_INDEX


# --- Строгая, изолированная от модели проверка ignore_index ---

def test_cross_entropy_ignores_masked_positions_regardless_of_logits():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 10)
    targets = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])

    logits_different_at_masked = logits.clone()
    logits_different_at_masked[:, :2, :] = torch.randn(1, 2, 10) * 1000

    loss_original = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1), ignore_index=IGNORE_INDEX)
    loss_modified = F.cross_entropy(logits_different_at_masked.reshape(-1, 10), targets.reshape(-1), ignore_index=IGNORE_INDEX)

    assert loss_original.item() == loss_modified.item()


def test_collate_sft_batch_padding_and_mask():
    examples = [
        ([1, 2, 3], [IGNORE_INDEX, IGNORE_INDEX, 3]),
        ([4, 5, 6, 7, 8], [IGNORE_INDEX, 6, 7, 8, IGNORE_INDEX]),
    ]
    input_ids, labels, attention_mask = collate_sft_batch(examples, pad_token_id=0)

    assert input_ids.shape == (2, 5)
    assert input_ids[0].tolist() == [1, 2, 3, 0, 0]
    assert labels[0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 3, IGNORE_INDEX, IGNORE_INDEX]
    assert attention_mask[0].tolist() == [1, 1, 1, 0, 0]
    assert attention_mask[1].tolist() == [1, 1, 1, 1, 1]


def test_sft_train_step_runs_and_produces_finite_loss():
    tokenizer = _make_test_tokenizer()
    torch.manual_seed(0)
    model = Transformer(
        vocab_size=tokenizer.vocab_size + 10, num_layer=2, hidden_size=16, head_dim=4,
        num_heads=4, num_kv_heads=2, use_qk_norm=True, qk_norm_eps=1e-6,
        rope_theta=10000.0, max_position_embeddings=64,
        intermediate_size=32, norm_eps=1e-6,
    )

    examples_raw = [
        {"messages": [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "this is a test"}]},
        {"messages": [{"role": "user", "content": "напиши функцию"}, {"role": "assistant", "content": "def foo(): return 1"}]},
    ]
    tokenized = [tokenize_sft_example(tokenizer, ex, SPECIAL_TOKENS, max_len=64) for ex in examples_raw]
    tokenized = [t for t in tokenized if t is not None]

    input_ids, labels, attention_mask = collate_sft_batch(tokenized, pad_token_id=tokenizer.pad_token_id)

    loss = sft_train_step(model, input_ids, labels, attention_mask, grad_accum_steps=1)
    assert loss == loss
    assert loss > 0

    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert has_grad


# --- format_prompt_for_generation ---

def test_format_prompt_for_generation_ends_with_assistant_start():
    messages = [{"role": "user", "content": "hello"}]
    prompt = format_prompt_for_generation(messages, SPECIAL_TOKENS)
    assert prompt.endswith(SPECIAL_TOKENS["assistant_start"])
    assert SPECIAL_TOKENS["user_start"] in prompt
    assert "hello" in prompt


def test_format_prompt_for_generation_includes_history():
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    prompt = format_prompt_for_generation(messages, SPECIAL_TOKENS)
    assert "first question" in prompt
    assert "first answer" in prompt
    assert "second question" in prompt
    assert prompt.endswith(SPECIAL_TOKENS["assistant_start"])
