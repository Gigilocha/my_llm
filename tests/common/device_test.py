# tests/test_device.py
import pytest
import torch
from src.common.device import (
    resolve_device,
    is_distributed,
    get_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
)


# --- resolve_device ---

def test_resolve_device_auto_returns_valid_device():
    result = resolve_device("auto")
    assert result in ("cuda", "cpu")
    assert result == ("cuda" if torch.cuda.is_available() else "cpu")


def test_resolve_device_explicit_cpu_stays_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda") == "cpu"


def test_resolve_device_cuda_stays_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == "cuda"


# --- distributed-хелперы без torchrun (обычный однопроцессный запуск) ---

def test_is_distributed_false_without_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert is_distributed() is False


def test_defaults_are_single_process(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert get_rank() == 0
    assert get_local_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True


def test_is_distributed_true_with_torchrun_env(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert is_distributed() is True
    assert get_rank() == 1
    assert get_world_size() == 4
    assert is_main_process() is False
