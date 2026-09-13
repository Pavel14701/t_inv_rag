"""Tests for StrategyRegistry: register, query, delete (TZ-02 п.3.2)."""

from __future__ import annotations

import pytest

from strategies.src.application.strategy import (
    Metrics,
    Strategy,
    StrategyRegistry,
)


@pytest.fixture
def registry(tmp_path) -> StrategyRegistry:
    """A fresh registry in a temp directory."""
    return StrategyRegistry(tmp_path / "data")


@pytest.fixture
def valid_strategy() -> Strategy:
    return Strategy(
        id="strat-1",
        name="RSI entry",
        description="Simple RSI entry",
        dsl_entry="rsi.value < 30",
        dsl_exit="close > open",
    )


class TestRegistry:
    def test_register_and_get(
        self,
        registry: StrategyRegistry,
        valid_strategy: Strategy,
        manifest: dict,
    ) -> None:
        stored = registry.register(valid_strategy, manifest)
        assert stored.manifest_hash != ""  # auto-pinned
        loaded = registry.get("strat-1")
        assert loaded is not None
        assert loaded.id == "strat-1"
        assert loaded.dsl_entry == "rsi.value < 30"

    def test_register_invalid_raises(
        self, registry: StrategyRegistry, manifest: dict
    ) -> None:
        bad = Strategy(
            id="bad", name="x", description="x", dsl_entry="stoch_rsi < 20"
        )
        with pytest.raises(ValueError, match="failed validation"):
            registry.register(bad, manifest)

    def test_register_unknown_indicator_raises(
        self, registry: StrategyRegistry, manifest: dict
    ) -> None:
        bad = Strategy(
            id="bad", name="x", description="x", dsl_entry="close >"
        )
        with pytest.raises(ValueError, match="failed validation"):
            registry.register(bad, manifest)

    def test_get_missing_returns_none(
        self, registry: StrategyRegistry
    ) -> None:
        assert registry.get("nonexistent") is None

    def test_list_and_delete(
        self,
        registry: StrategyRegistry,
        valid_strategy: Strategy,
        manifest: dict,
    ) -> None:
        registry.register(valid_strategy, manifest)
        assert len(registry.list()) == 1
        assert registry.delete("strat-1") is True
        assert registry.delete("strat-1") is False  # already gone
        assert registry.list() == []

    def test_by_manifest_hash(
        self,
        registry: StrategyRegistry,
        valid_strategy: Strategy,
        manifest: dict,
    ) -> None:
        stored = registry.register(valid_strategy, manifest)
        matches = registry.by_manifest(stored.manifest_hash)
        assert len(matches) == 1
        assert registry.by_manifest("wrong_hash") == []

    def test_metrics_roundtrip(
        self,
        registry: StrategyRegistry,
        valid_strategy: Strategy,
        manifest: dict,
    ) -> None:
        strategy = Strategy(
            **{
                **valid_strategy.to_dict(),
                "metrics": Metrics(profit_factor=1.8, sharpe=1.2, n_trades=42),
            }
        )
        registry.register(strategy, manifest)
        loaded = registry.get("strat-1")
        assert loaded is not None
        assert loaded.metrics is not None
        assert loaded.metrics.profit_factor == pytest.approx(1.8)
        assert loaded.metrics.n_trades == 42
