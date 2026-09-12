"""Risk package (TZ-11): config-driven deterministic risk engine.

Rules and limits are *data* (configs/risk.yaml); the engine only
interprets them. Hard invariants (no order without approve, size==0 on
empty capital) live in code and tests, not in the config.
"""

from .config import (
    InstrumentOverride,
    PortfolioConfig,
    RiskConfig,
    RiskSeverity,
    RuleInstance,
    load_config,
    validate_config,
)
from .engine import Decision, PortfolioState, Signal, check


__all__ = [
    "Decision",
    "InstrumentOverride",
    "PortfolioConfig",
    "PortfolioState",
    "RiskConfig",
    "RiskSeverity",
    "RuleInstance",
    "Signal",
    "check",
    "load_config",
    "validate_config",
]
