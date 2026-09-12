"""Tests for device resolution (TZ-06 п.11)."""

import pytest

from ai.src.config import ComputeConfig
from ai.src.device import resolve_infer_device, resolve_train_device


def test_train_cpu_explicit():
    """Explicit cpu backend resolves to cpu device."""
    handle = resolve_train_device(ComputeConfig(train_backend="cpu"))
    assert handle.kind == "cpu"


def test_train_rejects_non_cuda_backends():
    """Training backends limited to cuda/cpu."""
    with pytest.raises(ValueError):
        resolve_train_device(ComputeConfig(train_backend="directml"))
    with pytest.raises(ValueError):
        resolve_train_device(ComputeConfig(train_backend="vulkan"))


def test_train_cuda_unavailable_raises():
    """Explicit cuda without hardware raises RuntimeError."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA available on this machine")
    with pytest.raises(RuntimeError):
        resolve_train_device(ComputeConfig(train_backend="cuda"))


def test_infer_vulkan_is_alias_for_onnx_directml():
    """'vulkan' resolves to the DX12 (DirectML) path."""
    cfg = ComputeConfig(infer_backend="vulkan")
    try:
        handle = resolve_infer_device(cfg)
    except RuntimeError as exc:
        pytest.skip(f"gpu extra not installed: {exc}")
    assert handle.kind == "directml"


def test_infer_unknown_backend_raises():
    """Unknown infer backend raises ValueError."""
    with pytest.raises(ValueError):
        resolve_infer_device(ComputeConfig(infer_backend="tpu"))


def test_infer_cpu_explicit():
    """Explicit cpu backend resolves to cpu device."""
    handle = resolve_infer_device(ComputeConfig(infer_backend="cpu"))
    assert handle.kind == "cpu"
