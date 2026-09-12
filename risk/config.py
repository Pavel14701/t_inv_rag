"""Typed risk configuration with strict validation (TZ-11 item 3.1-3.2).

Rules and limits are data. Everything that influences a decision is read
from the config at startup - no magic numbers in the engine. Unknown keys
or invalid values raise during ``validate_config()`` instead of being
silently ignored.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class RiskSeverity(Enum):
    """Severity of a rule. v1 supports only ENFORCE (TZ-11 item 3.3)."""

    ENFORCE = "enforce"
    # ``warn`` is intentionally rejected until the Baseline-gate
    # (TZ-04 item 4.6.1) is passed - soft mode would blur determinism.


class ConfigValidationError(ValueError):
    """Raised when a risk config violates the declared schema."""


def _severity_from(v: Any) -> RiskSeverity:
    if not isinstance(v, str):
        raise ConfigValidationError(f"severity must be a string, got {v!r}")
    if v != RiskSeverity.ENFORCE.value:
        raise ConfigValidationError(
            f"severity={v!r} is not allowed in v1; only 'enforce'"
        )
    return RiskSeverity.ENFORCE


def _reject_unknown(
    raw: dict[str, Any], allowed: set[str], where: str
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigValidationError(
            f"unknown key(s) in {where}: {sorted(unknown)} "
            f"(allowed: {sorted(allowed)})"
        )


@dataclass(frozen=True, slots=True)
class RuleInstance:
    """One configured rule in the engine pipeline (TZ-11 item 3.3)."""

    name: str
    active: bool
    severity: RiskSeverity
    params: dict[str, Any]  # data only; never code

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "RuleInstance":
        """Parse and validate a single rule entry."""
        _reject_unknown(raw, {"name", "active", "severity", "params"}, "rule")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigValidationError(
                "rule 'name' is required and must be a non-empty string"
            )
        return RuleInstance(
            name=name,
            active=bool(raw.get("active", True)),
            severity=_severity_from(
                raw.get("severity", RiskSeverity.ENFORCE.value)
            ),
            params=dict(raw.get("params", {}) or {}),
        )


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Ordered list of rules applied on every signal."""

    rules: tuple[RuleInstance, ...]

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "EngineConfig":
        """Parse the ordered rule pipeline."""
        _reject_unknown(raw, {"rules"}, "engine")
        rules_raw = raw.get("rules")
        if not isinstance(rules_raw, list):
            raise ConfigValidationError("engine.rules must be a list")
        return EngineConfig(
            rules=tuple(RuleInstance.from_dict(r) for r in rules_raw)
        )


@dataclass(frozen=True, slots=True)
class InstrumentOverride:
    """Per-instrument overrides that shadow portfolio-level limits."""

    max_capital_pct: float | None = None
    max_units: float | None = None

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "InstrumentOverride":
        """Parse one instrument override block."""
        _reject_unknown(raw, {"max_capital_pct", "max_units"}, "instrument")
        mcp = raw.get("max_capital_pct")
        mu = raw.get("max_units")
        if mcp is not None and not isinstance(mcp, (int, float)):
            raise ConfigValidationError("max_capital_pct must be numeric")
        if mu is not None and not isinstance(mu, (int, float)):
            raise ConfigValidationError("max_units must be numeric")
        return InstrumentOverride(
            max_capital_pct=float(mcp) if mcp is not None else None,
            max_units=float(mu) if mu is not None else None,
        )


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """Portfolio-level limits plus per-instrument overrides."""

    max_capital_pct: float = 0.1
    max_units: float | None = None
    instruments: dict[str, InstrumentOverride] = field(default_factory=dict)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "PortfolioConfig":
        """Parse portfolio-level limits and overrides."""
        _reject_unknown(
            raw,
            {"max_capital_pct", "max_units", "instruments"},
            "portfolio",
        )
        mcp = raw.get("max_capital_pct", 0.1)
        if not isinstance(mcp, (int, float)):
            raise ConfigValidationError(
                "portfolio.max_capital_pct must be numeric"
            )
        instruments = raw.get("instruments") or {}
        if not isinstance(instruments, dict):
            raise ConfigValidationError(
                "portfolio.instruments must be a mapping"
            )
        mu_raw = raw.get("max_units")
        return PortfolioConfig(
            max_capital_pct=float(mcp),
            max_units=float(mu_raw) if mu_raw is not None else None,
            instruments={
                k: InstrumentOverride.from_dict(v)
                for k, v in instruments.items()
            },
        )


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Root risk configuration (mirrors configs/risk.yaml)."""

    engine: EngineConfig
    portfolio: PortfolioConfig

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "RiskConfig":
        """Parse the root config from a raw mapping."""
        _reject_unknown(raw, {"engine", "portfolio"}, "risk config")
        if "engine" not in raw:
            raise ConfigValidationError("missing top-level 'engine'")
        return RiskConfig(
            engine=EngineConfig.from_dict(raw["engine"]),
            portfolio=PortfolioConfig.from_dict(raw.get("portfolio", {})),
        )


def validate_config(cfg: RiskConfig) -> None:
    """Validate a loaded config: known rule names, bounded numerics.

    Rule-name check imports the registry lazily to avoid a circular import
    between ``risk.config`` and ``risk.rules``.
    """
    from .rules import registered_names

    known = registered_names()
    for rule in cfg.engine.rules:
        if rule.name not in known:
            raise ConfigValidationError(
                f"unknown rule name {rule.name!r}; registered: {sorted(known)}"
            )
    if not (0.0 < cfg.portfolio.max_capital_pct <= 1.0):
        raise ConfigValidationError(
            "portfolio.max_capital_pct must be in (0, 1], got "
            f"{cfg.portfolio.max_capital_pct}"
        )


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in a.items()}
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# Declared RISK_* env overrides (strict, schema-known only). Anything else
# under the RISK_ prefix is rejected eagerly so a typo is never silent.
_ENV_KEYS = frozenset({"RISK_MAX_CAPITAL_PCT", "RISK_MAX_UNITS"})


def _reject_unknown_env(env: dict[str, str]) -> None:
    unknown = {k for k in env if k.startswith("RISK_")} - _ENV_KEYS
    if unknown:
        raise ConfigValidationError(
            f"unknown RISK_* env var(s): {sorted(unknown)}; "
            f"declared: {sorted(_ENV_KEYS)}"
        )


def _apply_env(cfg: RiskConfig, env: dict[str, str]) -> RiskConfig:
    """Return a RiskConfig overridden by declared RISK_* env vars.

    Runtime-effective overrides keep config-driven behavior without code
    edits (TZ-11 item 3.2). Returns the same config when no override applies.
    """
    mcp = env.get("RISK_MAX_CAPITAL_PCT")
    mu = env.get("RISK_MAX_UNITS")
    if not mcp and not mu:
        return cfg
    portfolio = cfg.portfolio
    new_kwargs: dict[str, Any] = {}
    if mcp:
        try:
            new_kwargs["max_capital_pct"] = float(mcp)
        except ValueError as exc:
            raise ConfigValidationError(
                f"RISK_MAX_CAPITAL_PCT must be a float, got {mcp!r}"
            ) from exc
    if mu:
        try:
            new_kwargs["max_units"] = float(mu)
        except ValueError as exc:
            raise ConfigValidationError(
                f"RISK_MAX_UNITS must be a float, got {mu!r}"
            ) from exc
    new_portfolio = PortfolioConfig(
        max_capital_pct=new_kwargs.get(
            "max_capital_pct", portfolio.max_capital_pct
        ),
        max_units=new_kwargs.get("max_units", portfolio.max_units),
        instruments=portfolio.instruments,
    )
    overridden = RiskConfig(engine=cfg.engine, portfolio=new_portfolio)
    validate_config(overridden)
    return overridden


def load_config(
    *,
    path: str | os.PathLike | None = None,
    data: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    base: Path | None = None,
) -> RiskConfig:
    """Load a RiskConfig from defaults, user file and env (TZ-11 item 3.2).

    Layering: ``data``/default file -> ``path`` (user file, deep merge)
    -> declared ``RISK_*`` env overrides. Every layer is validated.
    """
    env_map = dict(os.environ if env is None else env)
    _reject_unknown_env(env_map)

    raw: dict[str, Any] = {}
    if data is None:
        base = base or Path(__file__).resolve().parents[1] / "configs"
        default_file = base / "risk.yaml"
        if default_file.exists():
            raw = (
                yaml.safe_load(default_file.read_text(encoding="utf-8")) or {}
            )
    elif isinstance(data, dict):
        raw = _deep_merge(raw, data)

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigValidationError(f"user config not found: {p}")
        user_raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, user_raw)

    cfg = RiskConfig.from_dict(raw)
    validate_config(cfg)
    return _apply_env(cfg, env_map)
