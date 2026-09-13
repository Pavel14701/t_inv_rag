"""Compute backends: CUDA/CPU training, DX12 ("vulkan") inference.

Policy (TZ-06 item 11):
- training - CUDA or CPU only; non-CUDA training was dropped;
- inference - torch (cuda/cpu) or ONNX Runtime with the DirectML EP
  (DX12: works on AMD/Intel/NVIDIA; 'vulkan' is an alias of this path,
  because the pure Vulkan EP in ONNX Runtime is still experimental);
- GGUF/llama.cpp rejected: the custom EntryExitTransformer architecture
  is not supported by GGUF converters.

The DirectML path requires the optional dependency group ``gpu``
(``uv sync --extra gpu``) and is only available on Windows/DX12.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ComputeConfig


__all__ = (
    "DeviceHandle",
    "export_onnx",
    "resolve_infer_device",
    "resolve_train_device",
)

_TRAIN_BACKENDS = ("auto", "cuda", "cpu")
_INFER_BACKENDS = ("auto", "cuda", "cpu", "onnx_directml", "vulkan")


@dataclass(frozen=True, slots=True)
class DeviceHandle:
    """Resolved device plus a description of how tensors must be moved.

    Attributes:
        kind: 'cuda' | 'cpu' | 'directml'.
        torch_device: ``torch.device`` for cuda/cpu, None for directml
            (tensors are moved via the module-level ``torch_directml``
            device object instead).

    """

    kind: str
    torch_device: torch.device | None = None

    def to(self, tensor):
        """Move a torch tensor onto this device."""
        if self.kind == "directml":
            import torch_directml

            return tensor.to(torch_directml.device())
        return tensor.to(self.torch_device)  # type: ignore[union-attr]


def _auto_device(want_cuda: bool) -> DeviceHandle:
    import torch

    if want_cuda and torch.cuda.is_available():
        return DeviceHandle("cuda", torch.device("cuda"))
    return DeviceHandle("cpu", torch.device("cpu"))


def resolve_train_device(cfg: ComputeConfig) -> DeviceHandle:
    """Resolve device for training: cuda | cpu only.

    Args:
        cfg: Compute section of the config.

    Returns:
        Resolved :class:`DeviceHandle`.

    Raises:
        ValueError: If backend is not allowed for training.
        RuntimeError: If 'cuda' requested but unavailable.

    """
    backend = cfg.train_backend
    if backend not in _TRAIN_BACKENDS:
        raise ValueError(
            f"train_backend must be one of {_TRAIN_BACKENDS}, "
            f"got {backend!r}; non-CUDA training is not "
            f"supported (TZ-06 item 11)"
        )
    if backend == "cpu":
        import torch

        return DeviceHandle("cpu", torch.device("cpu"))
    handle = _auto_device(want_cuda=True)
    if backend == "cuda" and handle.kind != "cuda":
        raise RuntimeError(
            "train_backend='cuda' requested, but CUDA is not available"
        )
    return handle


def resolve_infer_device(cfg: ComputeConfig) -> DeviceHandle:
    """Resolve device for inference: cuda | cpu | directml ('vulkan').

    Args:
        cfg: Compute section of the config.

    Returns:
        Resolved :class:`DeviceHandle`.

    Raises:
        ValueError: If backend is unknown.
        RuntimeError: If the requested backend is unavailable.

    """
    backend = cfg.infer_backend
    if backend not in _INFER_BACKENDS:
        raise ValueError(
            f"infer_backend must be one of {_INFER_BACKENDS}, got {backend!r}"
        )
    if backend == "vulkan":  # DX12 alias, see ComputeConfig docstring
        backend = "onnx_directml"
    if backend == "cpu":
        import torch

        return DeviceHandle("cpu", torch.device("cpu"))
    if backend == "onnx_directml":
        try:
            import torch_directml  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "infer_backend='onnx_directml' requires the optional "
                "'gpu' dependency group: uv sync --extra gpu"
            ) from exc
        return DeviceHandle("directml")
    handle = _auto_device(want_cuda=True)
    if backend == "cuda" and handle.kind != "cuda":
        raise RuntimeError(
            "infer_backend='cuda' requested, but CUDA is not available"
        )
    return handle


def export_onnx(model, sample_inputs: tuple, cfg: ComputeConfig) -> Path:
    """Export the model to ONNX for non-torch inference backends.

    Args:
        model: ``EntryExitTransformer`` in eval mode.
        sample_inputs: Tuple of example positional inputs matching forward().
        cfg: Compute section (uses ``onnx_export_dir``).

    Returns:
        Path to the written ``.onnx`` file.

    """
    import torch

    out_dir = Path(cfg.onnx_export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "entry_exit_transformer.onnx"
    torch.onnx.export(
        model,
        sample_inputs,
        str(out_path),
        input_names=[f"input_{i}" for i in range(len(sample_inputs))],
        output_names=["action_logits", "outcome_logits", "pattern_logits"],
        dynamic_axes={
            name: {0: "batch", 1: "time"}
            for name in ("input_0", "input_1", "input_2", "input_3", "input_4")
        },
        opset_version=17,
    )
    return out_path
