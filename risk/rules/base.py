"""Rule base types and the name -> implementation registry (TZ-11 item 3.3).

Concrete rules live in :mod:`risk.rules.impl` and self-register via
:func:`register`. The engine only resolves names from this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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


_REGISTRY: dict[str, Callable[..., RuleResult]] = {}


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
