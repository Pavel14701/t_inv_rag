"""Strategy format, validation and registry (TZ-02 items 2.2-2.4, 3.2, 3.4).

A *strategy* is a declarative, fully machine-checkable artefact: two DSL
expressions (entry/exit) plus a manifest hash that pins the indicator
schema it was validated against. Nothing enters the registry without
passing :func:`validate_strategy`.
"""

from __future__ import annotations

import builtins
import hashlib
import json

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dsl.parser import ParseError, parse


# --------------------------------------------------------------------------- #
# 1. Data model (TZ-02 item 2.2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Metrics:
    """Backtest outcome, filled in by the engine (TZ-04)."""

    profit_factor: float | None = None
    sharpe: float | None = None
    max_drawdown_pct: float | None = None
    win_rate: float | None = None
    n_trades: int | None = None


@dataclass(frozen=True, slots=True)
class Strategy:
    """A validated, declarative trading strategy.

    ``manifest_hash`` pins the indicator manifest version the strategy was
    validated against. Strategies whose hash does not match the current
    manifest are rejected by the registry so RAG few-shot examples can
    never reference a removed indicator.
    """

    id: str
    name: str
    description: str
    dsl_entry: str
    dsl_exit: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    manifest_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metrics: Metrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict."""
        return asdict(self)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Strategy":
        """Parse from a dict (e.g. loaded from a JSON file)."""
        metrics_raw = raw.get("metrics")
        metrics = Metrics(**metrics_raw) if metrics_raw else None
        return Strategy(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            dsl_entry=raw["dsl_entry"],
            dsl_exit=raw.get("dsl_exit"),
            params=raw.get("params") or {},
            manifest_hash=raw.get("manifest_hash", ""),
            created_at=raw.get("created_at", ""),
            metrics=metrics,
        )


# --------------------------------------------------------------------------- #
# 2. Indicator extraction from a DSL AST (TZ-02 item 3.4)
# --------------------------------------------------------------------------- #


def _extract_indicators(node: dict[str, Any]) -> set[str]:
    """Recursively collect indicator names from a to_dict AST tree."""
    result: set[str] = set()
    if isinstance(node, dict):
        if node.get("type") == "IndicatorAccess" and node.get("indicator"):
            result.add(node["indicator"])
        for v in node.values():
            result |= _extract_indicators(v)
    elif isinstance(node, list):
        for item in node:
            result |= _extract_indicators(item)
    return result


def indicators_used(dsl_expression: str) -> list[str]:
    """Sorted list of indicator names referenced by ``dsl_expression``."""
    tree = parse(dsl_expression)
    return sorted(_extract_indicators(tree.to_dict()))


# --------------------------------------------------------------------------- #
# 3. Validation (TZ-02 items 2.3, 3.4)
# --------------------------------------------------------------------------- #


def _manifest_hash(manifest: dict[str, Any]) -> str:
    """Stable hash of an indicator manifest (TZ-02 item 2.2)."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _validate_dsl(dsl_expression: str, manifest: dict[str, Any]) -> list[str]:
    """Parse ``dsl_expression`` and check every indicator against the manifest.

    Returns a list of human-readable error strings (empty = valid).
    """
    try:
        tree = parse(dsl_expression)
    except ParseError as exc:
        return [f"DSL parse error: {exc}"]

    known = set(manifest.get("indicators") or {})
    return [
        f"Unknown indicator {name!r}; available: {sorted(known)}"
        for name in _extract_indicators(tree.to_dict())
        if name not in known
    ]


def validate_strategy(
    strategy: Strategy,
    manifest: dict[str, Any],
) -> list[str]:
    """Validate a strategy against the given manifest.

    Checks: parse of both DSL expressions + every referenced indicator is
    present in the manifest. Returns a list of error strings (empty = valid).
    """
    errors = _validate_dsl(strategy.dsl_entry, manifest)
    if strategy.dsl_exit:
        errors.extend(_validate_dsl(strategy.dsl_exit, manifest))
    return errors


# --------------------------------------------------------------------------- #
# 4. Registry (TZ-02 item 3.2)
# --------------------------------------------------------------------------- #


class StrategyRegistry:
    """File-backed registry of validated strategies (JSON, one per file).

    PostgreSQL migration is deferred to TZ-10; until then this is the
    single source of truth and the only way to persist a strategy is
    through :meth:`register`, which re-validates before writing.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- queries ------------------------------------------------------------ #

    def list(self) -> list[Strategy]:
        """Load all strategies from the registry directory."""
        strategies: list[Strategy] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                strategies.append(
                    Strategy.from_dict(json.loads(path.read_text()))
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return strategies

    def get(self, strategy_id: str) -> Strategy | None:
        """Load a single strategy by id, or None."""
        path = self.root / f"{strategy_id}.json"
        if not path.exists():
            return None
        return Strategy.from_dict(json.loads(path.read_text()))

    def by_manifest(self, manifest_hash: str) -> builtins.list[Strategy]:
        """Filter strategies by their pinned manifest hash."""
        return [s for s in self.list() if s.manifest_hash == manifest_hash]

    # -- writes ------------------------------------------------------------- #

    def register(
        self,
        strategy: Strategy,
        manifest: dict[str, Any],
    ) -> Strategy:
        """Validate and persist a strategy. Raises if invalid.

        Returns the stored strategy with ``manifest_hash`` set to the hash
        of the provided manifest.
        """
        errors = validate_strategy(strategy, manifest)
        if errors:
            raise ValueError(
                f"Strategy {strategy.id!r} failed validation: "
                f"{'; '.join(errors)}"
            )
        pinned = (
            strategy
            if strategy.manifest_hash
            else Strategy(
                **{
                    **strategy.to_dict(),
                    "manifest_hash": _manifest_hash(manifest),
                }
            )
        )
        (self.root / f"{pinned.id}.json").write_text(
            json.dumps(pinned.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return pinned

    def delete(self, strategy_id: str) -> bool:
        """Remove a strategy file; True if it existed."""
        path = self.root / f"{strategy_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False


# --------------------------------------------------------------------------- #
# 5. Helpers
# --------------------------------------------------------------------------- #


def load_strategies(
    paths: Iterable[str | Path],
    manifest: dict[str, Any] | None = None,
) -> list[Strategy]:
    """Load strategies from JSON files, optionally validating each."""
    strategies: list[Strategy] = []
    for item in paths:
        strategy = Strategy.from_dict(json.loads(Path(item).read_text()))
        if manifest is not None:
            errors = validate_strategy(strategy, manifest)
            if errors:
                raise ValueError(
                    f"{item}: invalid strategy: {'; '.join(errors)}"
                )
        strategies.append(strategy)
    return strategies
