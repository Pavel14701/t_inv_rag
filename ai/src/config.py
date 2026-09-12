"""YAML-конфигурация ai-модуля.

Единый источник параметров (TZ-06 п.10): риск-менеджмент/лейблы, архитектура
модели, обучение и вычислительный бэкенд. Значения по умолчанию = прежние
захардкоженные константы, поэтому существующий код ведёт себя идентично.

Конфиг протаскивается аргументами функций; глобального состояния нет.
"""

from __future__ import annotations

import random

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml


__all__ = (
    'AIConfig',
    'ComputeConfig',
    'ModelConfig',
    'RiskConfig',
    'TrainingConfig',
    'load_config',
    'risk_kwargs',
    'set_seed',
)

DEFAULT_CONFIG_PATH = Path('configs/ai.yaml')


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Риск-параметры генерации лейблов (бывший хардкод features.py)."""

    atr_period: int = 14
    atr_floor: float = 1.0e-6
    tp_atr_multiplier: float = 2.0
    sl_atr_multiplier: float = 1.5
    min_rr: float = 1 / 3
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    max_bars_hold: int = 20
    use_r_multiple: bool = False
    use_structure_filter: bool = False
    trend_filter: str | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Архитектура EntryExitTransformer."""

    seq_len: int = 128
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 1024
    max_ob_seq_len: int = 256
    n_action_classes: int = 3
    outcome_mode: str = 'binary'
    n_outcome_classes: int = 2
    n_patterns: int = 10
    ob_embedding_dim: int = 32
    atr_global: float = 1.0
    close_idx: int = 3


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Параметры обучения и self-training."""

    epochs: int = 10
    batch_size: int = 16
    lr: float = 1.0e-4
    lambda_outcome: float = 0.3
    lambda_pattern: float = 0.1
    val_split: float = 0.2
    patience: int = 3
    class_weight: bool = True
    num_rounds: int = 3
    action_threshold: float = 0.9
    outcome_threshold: float = 0.8


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    """Вычислительный бэкенд (TZ-06 п.11).

    train_backend: только cuda | cpu (не-CUDA обучение отброшено).
    infer_backend: cuda | cpu | onnx_directml; 'vulkan' — алиас
    onnx_directml (DX12: те же AMD/Intel/NVIDIA карты; чистый Vulkan
    и GGUF отвергнуты — кастомная архитектура не поддерживается
    llama.cpp, backward на Vulkan не существует).
    """

    train_backend: str = 'auto'
    infer_backend: str = 'auto'
    device_id: int = 0
    onnx_export_dir: str = 'runs/onnx'


@dataclass(frozen=True, slots=True)
class AIConfig:
    """Корневой конфиг ai-модуля."""

    seed: int = 42
    risk: RiskConfig = field(default_factory=RiskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)


def _build(cls: type, data: dict[str, Any] | None):
    """Create dataclass ``cls`` from dict, ignoring unknown keys."""
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path | None = None) -> AIConfig:
    """Load configuration from YAML; missing keys fall back to defaults.

    Args:
        path: Path to YAML file. If None, ``configs/ai.yaml`` is used when
            present; otherwise pure defaults.

    Returns:
        Populated :class:`AIConfig`.

    Raises:
        ValueError: If the YAML root is not a mapping.

    """
    file = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if file.exists():
        with open(file, encoding='utf-8') as fh:
            loaded = yaml.safe_load(fh)
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f'{file}: root must be a mapping')
            data = loaded
    return AIConfig(
        seed=data.get('seed', 42),
        risk=_build(RiskConfig, data.get('risk')),
        model=_build(ModelConfig, data.get('model')),
        training=_build(TrainingConfig, data.get('training')),
        compute=_build(ComputeConfig, data.get('compute')),
    )


def set_seed(seed: int) -> None:
    """Seed random, numpy and torch (if importable) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - hardware dependent
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch optional in some contexts
        pass


def risk_kwargs(risk: RiskConfig) -> dict[str, Any]:
    """Map :class:`RiskConfig` onto ``generate_labels_from_strategy`` kwargs.

    Returns:
        Dict with keys matching the label generator signature, so callers
        can do ``generate_labels_from_strategy(
        df, obs, **risk_kwargs(cfg.risk))``.

    """
    return {
        'min_rr': risk.min_rr,
        'use_r_multiple': risk.use_r_multiple,
        'use_structure_filter': risk.use_structure_filter,
        'trend_filter': risk.trend_filter,
        'commission_pct': risk.commission_pct,
        'slippage_pct': risk.slippage_pct,
        'max_bars_hold': risk.max_bars_hold,
    }


def with_overrides(
    cfg: AIConfig, **section_overrides: dict[str, Any]
) -> AIConfig:
    """Return a copy of ``cfg`` with per-section field overrides applied.

    Example: ``with_overrides(cfg, training={'epochs': 3})``.

    """
    updates: dict[str, Any] = {}
    for section_name, overrides in section_overrides.items():
        section = getattr(cfg, section_name)
        updates[section_name] = replace(section, **overrides)
    return replace(cfg, **updates)