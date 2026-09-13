# tests/test_sft_dataset.py
import pytest

from src.common.config import SFTSource
from src.data.sft_dataset import _extract_sft_example, _peek_and_validate_fields


def test_extract_example_from_messages_format():
    source = SFTSource(dataset_name="x", messages_field="messages")
    doc = {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ], "unrelated_field": 123}

    example = _extract_sft_example(doc, source)
    assert example == {"messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]}


def test_extract_example_from_flat_format():
    source = SFTSource(dataset_name="x", instruction_field="instruction", output_field="output", system_field="system")
    doc = {"instruction": "hello", "output": "hi there", "system": "be nice"}

    example = _extract_sft_example(doc, source)
    assert example["messages"][0] == {"role": "system", "content": "be nice"}
    assert example["messages"][1] == {"role": "user", "content": "hello"}
    assert example["messages"][2] == {"role": "assistant", "content": "hi there"}


def test_extract_example_flat_format_with_input_field():
    source = SFTSource(dataset_name="x", instruction_field="instruction", input_field="input", output_field="output")
    doc = {"instruction": "translate this", "input": "hello world", "output": "привет мир"}

    example = _extract_sft_example(doc, source)
    user_message = next(m for m in example["messages"] if m["role"] == "user")
    assert "translate this" in user_message["content"]
    assert "hello world" in user_message["content"]


def test_extract_example_returns_none_for_empty_fields():
    source = SFTSource(dataset_name="x", instruction_field="instruction", output_field="output")
    assert _extract_sft_example({"instruction": "", "output": "answer"}, source) is None
    assert _extract_sft_example({"instruction": "question", "output": ""}, source) is None


def test_extract_example_messages_format_skips_invalid_roles_and_empty_content():
    source = SFTSource(dataset_name="x", messages_field="messages")
    doc = {"messages": [
        {"role": "tool", "content": "irrelevant"},  # неизвестная роль — отбрасывается
        {"role": "user", "content": ""},  # пустой контент — отбрасывается
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "real answer"},
    ]}
    example = _extract_sft_example(doc, source)
    assert len(example["messages"]) == 2
    assert example["messages"][0]["content"] == "real question"


def test_extract_example_messages_format_without_assistant_returns_none():
    source = SFTSource(dataset_name="x", messages_field="messages")
    doc = {"messages": [{"role": "user", "content": "hello"}]}  # нет ответа ассистента
    assert _extract_sft_example(doc, source) is None


def test_peek_and_validate_fields_raises_clear_error_on_mismatch():
    source = SFTSource(dataset_name="x", messages_field="conversation")  # поля с таким именем нет
    stream = iter([{"messages": [{"role": "user", "content": "hi"}]}])

    with pytest.raises(ValueError, match="conversation"):
        _peek_and_validate_fields(stream, source)


def test_peek_and_validate_fields_passes_through_data_unchanged():
    source = SFTSource(dataset_name="x", messages_field="messages")
    original_docs = [{"messages": []}, {"messages": []}]
    stream = iter(original_docs)

    validated_stream = _peek_and_validate_fields(stream, source)
    assert list(validated_stream) == original_docs


# --- filter_field/filter_value: один датасет как несколько источников ---

def test_filter_field_requires_filter_value_together():
    with pytest.raises(ValueError):
        SFTSource(dataset_name="x", messages_field="messages", filter_field="supertag")
    with pytest.raises(ValueError):
        SFTSource(dataset_name="x", messages_field="messages", filter_values=["code"])
    # Оба вместе — ок
    SFTSource(dataset_name="x", messages_field="messages", filter_field="supertag", filter_values=["code"])


def test_cache_dir_disambiguated_by_cache_name():
    from src.data.sft_dataset import _local_dir_for_sft_source
    from pathlib import Path

    code_source = SFTSource(
        dataset_name="concretejungles/T-Wix-instag", messages_field="messages",
        filter_field="supertag", filter_values=["code"], cache_name="t-wix-code",
    )
    general_source = SFTSource(
        dataset_name="concretejungles/T-Wix-instag", messages_field="messages",
        filter_field="supertag", filter_values=["general"], cache_name="t-wix-general",
    )

    data_dir = Path("/tmp/fake_data")
    code_dir = _local_dir_for_sft_source(code_source, data_dir, "train")
    general_dir = _local_dir_for_sft_source(general_source, data_dir, "train")

    # Один и тот же dataset_name -> без cache_name пути бы совпали и затёрли
    # друг друга на диске; с cache_name они разные
    assert code_dir != general_dir
    assert "code" in str(code_dir)
    assert "general" in str(general_dir)
