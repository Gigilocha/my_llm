from unittest.mock import patch, MagicMock
from src.common.config import DataSource, PretrainData
from src.dataset import build_language_mix, build_pretrain_mix, extract_texts


def _fake_source(name="test/ds", weight=1.0):
    return DataSource(dataset_name=name, subset=None, split="train", weight=weight)


@patch("src.dataset.interleave_datasets")
@patch("src.dataset.load_source_stream")
def test_build_language_mix_normalizes_weights(mock_load, mock_interleave):
    mock_load.side_effect = lambda name, subset, split: MagicMock(name=name)
    sources = [_fake_source("a", weight=3.0), _fake_source("b", weight=1.0)]

    build_language_mix(sources, seed=42)

    _, kwargs = mock_interleave.call_args
    assert kwargs["probabilities"] == [0.75, 0.25]  # 3/(3+1), 1/(3+1)


@patch("src.dataset.interleave_datasets")
@patch("src.dataset.build_language_mix")
def test_build_pretrain_mix_uses_quantity_as_probabilities(mock_lang_mix, mock_interleave):
    mock_lang_mix.side_effect = lambda sources, seed: MagicMock()
    cfg = PretrainData(
        max_shard=10, seed=42, val_split_ratio=0.01,
        rus_quantity=60, en_quantity=20, code_quantity=20,
        rus_sources=[_fake_source()], en_sources=[_fake_source()], code_sources=[_fake_source()],
    )

    build_pretrain_mix(cfg)

    _, kwargs = mock_interleave.call_args
    assert kwargs["probabilities"] == [0.6, 0.2, 0.2]


def test_extract_texts_skips_empty_and_uses_fallback_field():
    fake_docs = [{"text": "hello"}, {"text": "  "}, {"content": "world"}, {}]
    result = list(extract_texts(fake_docs))
    assert result == ["hello", "world"]