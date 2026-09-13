"""Universal indicator mapper (TZ-03 wave 2): registry and TaProvider."""

from __future__ import annotations

import pytest

from dsl.context import Context
from dsl.interpreter import Interpreter
from dsl.parser import parse
from ta.src.provider import (
    BINDINGS,
    TaProvider,
    WarmupNotReady,
    build_manifest,
    registered_names,
)
from ta.src.registry import AUTO_REPORT, SKIP, SMOKE_SKIP


# Representative auto-derived indicators (one per group; fast subset of
# the full sweep - the full parametrized smoke is marked ``slow``).
FAST_SUBSET = ("macd", "ppo", "fisher", "brar", "kst", "ao")


def _ohlc(n: int = 500):
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(100, 10_000, n).astype(np.float64)
    return pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestRegistryCoverage:
    """Every ta indicator is either bound or explicitly skipped."""

    def test_all_indicators_exposed_or_reported(self) -> None:
        # 12 manual (wave-1 golden anchors + price passthroughs) + ~77 auto
        assert len(BINDINGS) > 80
        # manual golden anchors are never overwritten
        for name in ("ema", "sma", "rsi", "atr", "adx", "ott"):
            assert AUTO_REPORT[name] == "manual binding (golden anchor)"
        # every *_ind function is either bound, skipped or reported
        from ta.src import registry as reg

        for dsl_name, _fn in reg._iter_indicator_funcs():
            assert (
                dsl_name in BINDINGS or dsl_name in AUTO_REPORT
            ), f"{dsl_name} silently lost"

    def test_no_duplicate_sources_mismatch(self) -> None:
        """Auto bindings derive sources from data args."""
        assert BINDINGS["macd"].sources == ("close",)
        assert set(BINDINGS["brar"].sources) <= {
            "open", "high", "low", "close", "volume"
        }
        assert BINDINGS["brar"].sources

    def test_skip_table_reasons(self) -> None:
        assert "DataFrame" in SKIP["ichimoku"]
        assert AUTO_REPORT["scrsi"] == SKIP["scrsi"]


class TestManifestWave2:
    def test_manifest_includes_new_indicators(self) -> None:
        m = build_manifest()
        for name in ("macd", "supertrend", "fisher", "kst", "brar"):
            assert name in m["indicators"]

    def test_manifest_named_multi_outputs(self) -> None:
        macd = build_manifest()["indicators"]["macd"]
        assert macd["attributes"] == ["macd", "signal", "hist"]
        fisher = build_manifest()["indicators"]["fisher"]
        assert fisher["attributes"] == ["fisher", "signal"]

    def test_manifest_params_are_typed_and_clean(self) -> None:
        macd = build_manifest()["indicators"]["macd"]
        assert set(macd["parameters"]) == {"fast", "slow", "signal"}
        assert macd["parameters"]["fast"]["default"] == 12
        # engine args never leak into the manifest
        manifest = build_manifest()
        for name in registered_names():
            params = manifest["indicators"][name]["parameters"]
            engine_args = {"offset", "fillna", "use_talib", "trim"}
            assert not (engine_args & set(params))

    def test_generic_multi_output_names(self) -> None:
        st = build_manifest()["indicators"]["supertrend"]
        assert st["attributes"] == ["value", "out2", "out3", "out4"]


class TestEngineWave2:
    def test_multi_output_resolution(self, ) -> None:
        provider = TaProvider(_ohlc())
        provider.cursor = 200
        macd = provider.resolve("macd", {}, ["macd"], 0)
        sig = provider.resolve("macd", {}, ["signal"], 0)
        hist = provider.resolve("macd", {}, ["hist"], 0)
        assert all(isinstance(v, float) for v in (macd, sig, hist))
        assert hist == pytest.approx(macd - sig, abs=1e-9)

    def test_cache_key_covers_all_params(self) -> None:
        """macd(fast=12) and macd(fast=20) must not collide (wave 2 fix)."""
        provider = TaProvider(_ohlc())
        provider.cursor = 300
        fast12 = provider.resolve("macd", {"fast": 12}, ["macd"], 0)
        fast20 = provider.resolve("macd", {"fast": 20}, ["macd"], 0)
        assert fast12 != fast20
        # and the cached entry is reused for identical params
        again = provider.resolve("macd", {"fast": 12}, ["macd"], 0)
        assert again == fast12

    def test_warmup_contract(self) -> None:
        provider = TaProvider(_ohlc(60))
        provider.cursor = 20  # below macd slow default (26)
        with pytest.raises(WarmupNotReady):
            provider.resolve("macd", {}, ["macd"], 0)


class TestDslEndToEnd:
    def test_new_indicator_in_dsl_expression(self) -> None:
        provider = TaProvider(_ohlc())
        context = Context([provider])
        ast = parse("macd.hist > 0")
        provider.cursor = 300
        value = Interpreter(context).visit(ast)
        assert isinstance(value, bool)


@pytest.mark.parametrize("name", FAST_SUBSET)
@pytest.mark.slow
class TestSmokeSubset:
    """Fast subset: defaults resolve at a late bar without exceptions."""

    def test_resolve_default(self, name: str) -> None:
        provider = TaProvider(_ohlc())
        provider.cursor = 400
        value = provider.resolve(name, {}, [], 0)
        assert isinstance(value, float)


@pytest.mark.slow
class TestSmokeAll:
    """Full sweep over every registered binding (Numba cold compile)."""

    @pytest.mark.parametrize(
        "name",
        sorted(n for n in BINDINGS if n not in SMOKE_SKIP),
    )
    def test_resolve_default(self, name: str) -> None:
        provider = TaProvider(_ohlc())
        provider.cursor = 400
        value = provider.resolve(name, {}, [], 0)
        assert isinstance(value, float)

    def test_smoke_skip_documented(self) -> None:
        """Every excluded-from-smoke binding has a documented reason."""
        assert SMOKE_SKIP.keys() <= BINDINGS.keys()
        assert all(SMOKE_SKIP.values())
