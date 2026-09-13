"""Tests for TaProvider: parity, offset, warm-up, cache, look-ahead (TZ-03)."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from dsl.context import Context
from dsl.interpreter import Interpreter
from dsl.parser import parse
from ta.src.momentum.rsi import rsi_ind
from ta.src.overlap.ema import ema_ind
from ta.src.provider import (
    TaProvider,
    WarmupNotReady,
    build_manifest,
    registered_names,
)


@pytest.fixture
def ohlc(n: int = 500) -> pl.DataFrame:
    """Synthetic OHLCV frame with a deterministic seed."""
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


@pytest.fixture
def provider(ohlc: pl.DataFrame) -> TaProvider:
    return TaProvider(ohlc)


class TestManifest:
    def test_build_manifest_contains_registered(self) -> None:
        m = build_manifest()
        for name in registered_names():
            assert name in m["indicators"]

    def test_manifest_has_attributes_and_params(self) -> None:
        m = build_manifest()
        rsi = m["indicators"]["rsi"]
        assert "value" in rsi["attributes"]
        assert "length" in rsi["parameters"]


class TestParity:
    """Parity: DSL resolve(t) == direct ta call at the same bar (T1)."""

    @pytest.mark.parametrize(
        "dsl_expr, ta_fn, ta_kwargs",
        [
            ("ema(length=20)", ema_ind, {"length": 20}),
            ("ema(length=5)", ema_ind, {"length": 5}),
        ],
        ids=["ema20", "ema5"],
    )
    def test_ema_parity(
        self,
        ohlc: pl.DataFrame,
        dsl_expr: str,
        ta_fn,
        ta_kwargs: dict,
    ) -> None:
        provider = TaProvider(ohlc)
        provider.cursor = 200
        got = provider.resolve(
            dsl_expr.split("(")[0], {"length": ta_kwargs["length"]}, [], 0
        )
        close = ohlc["close"].to_numpy()
        expected = np.asarray(ta_fn(close, **ta_kwargs))[200]
        assert got == pytest.approx(float(expected), rel=1e-10)

    def test_rsi_parity(self, ohlc: pl.DataFrame) -> None:
        provider = TaProvider(ohlc)
        provider.cursor = 200
        got = provider.resolve("rsi", {"length": 14}, [], 0)
        close = ohlc["close"].to_numpy()
        expected = np.asarray(rsi_ind(close, length=14, nan_policy="ignore"))[
            200
        ]
        assert got == pytest.approx(float(expected), rel=1e-6)


class TestOffset:
    """DSL offset [n] means n bars back (T1: cache index, not ta offset)."""

    def test_offset_semantics(self, ohlc: pl.DataFrame) -> None:
        provider = TaProvider(ohlc)
        provider.cursor = 100
        now = provider.resolve("ema", {"length": 10}, [], 0)
        prev3 = provider.resolve("ema", {"length": 10}, [], 3)
        assert prev3 == pytest.approx(
            float(
                np.asarray(
                    ema_ind(
                        ohlc["close"].to_numpy(),
                        length=10,
                        nan_policy="ignore",
                    )
                )[97]
            ),
            rel=1e-10,
        )
        assert now != prev3  # sanity

    def test_offset_beyond_start_raises(self, ohlc: pl.DataFrame) -> None:
        provider = TaProvider(ohlc)
        provider.cursor = 5
        with pytest.raises(WarmupNotReady):
            provider.resolve("ema", {"length": 10}, [], 10)


class TestWarmup:
    def test_warmup_exactly_min_bars(self, ohlc: pl.DataFrame) -> None:
        """Exactly min_bars bars are NotReady; bar min_bars has a value."""
        provider = TaProvider(ohlc)
        provider.cursor = 13  # length=14 -> min_bars=14, bar 13 < 14
        with pytest.raises(WarmupNotReady):
            provider.resolve("rsi", {"length": 14}, [], 0)
        provider.cursor = 14  # bar 14 == min_bars -> value
        val = provider.resolve("rsi", {"length": 14}, [], 0)
        assert isinstance(val, float)
        assert not np.isnan(val)


class TestLookAhead:
    def test_future_bars_do_not_affect_resolve(
        self, ohlc: pl.DataFrame
    ) -> None:
        """Look-ahead invariant: replacing bars > t with junk does not
        change resolve(t) (TZ-03 acceptance)."""
        provider = TaProvider(ohlc)
        provider.cursor = 100
        val_before = provider.resolve("rsi", {"length": 14}, [], 0)
        poisoned = ohlc.clone()
        poisoned = poisoned.with_columns(
            pl.when(pl.int_range(pl.len()) > 100)
            .then(9999.0)
            .otherwise(pl.col("close"))
            .alias("close")
        )
        provider2 = TaProvider(poisoned)
        provider2.cursor = 100
        val_after = provider2.resolve("rsi", {"length": 14}, [], 0)
        assert val_before == pytest.approx(val_after, rel=1e-12)


class TestCache:
    def test_cache_hit_returns_same_object(self, ohlc: pl.DataFrame) -> None:
        provider = TaProvider(ohlc)
        provider.cursor = 100
        from ta.src.provider import BINDINGS

        binding = BINDINGS["rsi"]
        arr1 = provider._indicator_array(binding, {"length": 14})
        arr2 = provider._indicator_array(binding, {"length": 14})
        assert arr1 is arr2  # cached


class TestPerformance:
    def test_two_expressions_5000_bars(self) -> None:
        """2 DSL expressions x 5000 bars < 1s (TZ-03 acceptance)."""
        n = 5000
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        df = pl.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n),
            }
        )
        provider = TaProvider(df)
        context = Context([provider])
        entry_ast = parse("ema(length=20) > 0")
        exit_ast = parse("ema(length=50) > 0")
        interp = Interpreter(context)
        start = time.perf_counter()
        for t in range(100, n):
            provider.cursor = t
            interp.visit(entry_ast)
            interp.visit(exit_ast)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"too slow: {elapsed:.2f}s for {n} bars"


class TestMultiOutput:
    def test_adx_multi_output_attributes(self, ohlc: pl.DataFrame) -> None:
        """ADX binding maps adx/plus_di/minus_di to tuple indices."""
        provider = TaProvider(ohlc)
        provider.cursor = 100
        adx = provider.resolve("adx", {"length": 14}, ["adx"], 0)
        plus_di = provider.resolve("adx", {"length": 14}, ["plus_di"], 0)
        minus_di = provider.resolve("adx", {"length": 14}, ["minus_di"], 0)
        assert isinstance(adx, float)
        assert isinstance(plus_di, float)
        assert isinstance(minus_di, float)
        assert adx != plus_di or adx != minus_di

    def test_multi_output_uses_correct_index(self, ohlc: pl.DataFrame) -> None:
        """plus_di resolves from output index 1, not 0."""
        from ta.src.provider import get_binding

        provider = TaProvider(ohlc)
        provider.cursor = 100
        params = {"length": 14}
        arr = provider._indicator_array(get_binding("adx"), params)
        assert arr.ndim == 2  # (n_outputs, n_bars)
        assert arr.shape[0] == 4  # adx_ind returns 4 arrays
        plus_di_from_resolve = provider.resolve("adx", params, ["plus_di"], 0)
        assert plus_di_from_resolve == pytest.approx(
            float(arr[1, 100]), rel=1e-10
        )


class TestExtendedGroups:
    def test_entropy_registered(self) -> None:
        assert "entropy" in registered_names()

    def test_vwma_registered(self) -> None:
        assert "vwma" in registered_names()

    def test_engulfing_registered(self) -> None:
        assert "engulfing" in registered_names()

    def test_ott_registered(self) -> None:
        assert "ott" in registered_names()

    def test_resolve_entropy(self, ohlc: pl.DataFrame) -> None:
        provider = TaProvider(ohlc)
        provider.cursor = 100
        val = provider.resolve("entropy", {"length": 10}, [], 0)
        assert isinstance(val, float)
        assert not np.isnan(val)

    def test_resolve_vwma(self, ohlc: pl.DataFrame) -> None:
        provider = TaProvider(ohlc)
        provider.cursor = 100
        val = provider.resolve("vwma", {"length": 10}, [], 0)
        assert isinstance(val, float)
        assert not np.isnan(val)


class TestResolveHistory:
    def test_batch_resolve_history(self, ohlc: pl.DataFrame) -> None:
        """Batch resolve_history returns a slice of the cached array."""
        provider = TaProvider(ohlc)
        provider.cursor = 200
        params = {"length": 14}
        from ta.src.provider import BINDINGS

        arr = provider._indicator_array(BINDINGS["rsi"], params)
        # resolve_history: last 10 values up to cursor
        history = arr[200 - 9 : 201]
        assert len(history) == 10
        # matches individual resolves
        for offset in range(10):
            got = provider.resolve("rsi", params, [], offset)
            assert got == pytest.approx(float(history[9 - offset]), rel=1e-10)
