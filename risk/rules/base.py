"""Rule base types and the name -> implementation registry (TZ-11 item 3.3).

Concrete rules live in :mod:`risk.rules.impl` and self-register via
:func:`register`. The engine only resolves names from this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Outcome of a single rule. ``ok=False`` blocks the order."""

    ok: bool
    reason: str = ""
    size: float | None = None


@dataclass(slots=True)
class RuleState:
    """Runtime portfolio state the rules read (engine-owned, mutable)."""

    capital: float
    open_positions: int
    day_start_capital: float
    day_pnl: float
    peak_capital: float
    dd_pause_remaining: int = 0


_REGISTRY: dict[str, Callable[..., RuleResult]] = {}

# Declared parameter schema per rule (TZ-11 item 3.1): every key a rule
# reads, with its type. A nested dict value declares a sub-block of
# {sub_key: type} entries. validate_config() rejects any param key or type
# outside these schemas - a typo is never silently ignored.
PARAM_SCHEMAS: dict[str, dict[str, Any]] = {
    "require_stop_loss": {
        "tp": {"min_mult": float},
        "sl": {"min_mult": float},
    },
    "position_limit": {"max_capital_pct": float, "max_units": float},
    "max_positions": {"max_open": int},
    "daily_loss_limit": {"max_daily_loss_pct": float},
    "drawdown_stop": {"max_drawdown_pct": float, "pause_bars": int},
}


def register(name: str) -> Callable[[Callable], Callable]:
    """Class decorator helper: register a rule under ``name``."""

    def deco(fn: Callable[..., RuleResult]) -> Callable[..., RuleResult]:
        _REGISTRY[name] = fn
        return fn

    return deco


def registered_names() -> set[str]:
    """Names available to ``engine.rules`` config entries."""
    return set(_REGISTRY)


def get(name: str) -> Callable[..., RuleResult]:
    """Resolve a rule implementation by its config name."""
    return _REGISTRY[name]
