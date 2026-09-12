# -*- coding: utf-8 -*-
"""Shared helpers for candle pattern tests.

Provides:
- ohlc(): expand a list of (open, high, low, close) tuples into arrays
- FLAT / FLAT_BODY: neutral candles that trigger no pattern
- pattern_suite(): factory generating a uniform pytest suite for a cdl_*
  module (bullish/bearish detection, no-pattern, talib backend, read-only
  polars input, offset/fillna, polars wrapper).
"""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ....external import talib_available

# Neutral candle: small real WHITE body, modest shadows. Repeated identical
# white candles satisfy no classic black/white pattern condition.
FLAT = (99.7, 100.4, 99.3, 100.0)
# Neutral candle with a proportionally large body (doji-family modules:
# the tiny FLAT body would itself classify as a doji).
FLAT_BODY = (99.0, 100.5, 98.8, 100.2)


def ohlc(candles):
    """Expand list of (o, h, l, c) tuples into four float64 arrays."""
    o = np.array([c[0] for c in candles], dtype=np.float64)
    h = np.array([c[1] for c in candles], dtype=np.float64)
    low_ = np.array([c[2] for c in candles], dtype=np.float64)
    c = np.array([c[3] for c in candles], dtype=np.float64)
    return o, h, low_, c


def with_prefix(pattern_candles, n=4, flat=FLAT):
    """Prepend n flat candles; the pattern completes at the last bar."""
    return [flat] * n + list(pattern_candles)


def pattern_suite(
    name,
    fn,
    polars_fn=None,
    output_col=None,
    bull=None,
    bear=None,
    bull_value=100.0,
    bear_value=-100.0,
    bull_idx=None,
    bear_idx=None,
    talib_values=(0.0, 100.0, -100.0),
    extra=None,
    flat=FLAT,
    prefix_n=4,
):
    """Generate a uniform pytest test suite for a candle pattern module.

    Parameters
    ----------
    name : str
        Pattern name used in test ids (e.g. 'cdl_piercing').
    fn : callable
        Universal pattern function (open_, high, low, close, ...).
    polars_fn : callable, optional
        Module-level polars wrapper (df, ...) -> df.
    output_col : str, optional
        Column name added by polars_fn.
    bull, bear : list of (o, h, l, c), optional
        Candle sequences completing the pattern at the LAST bar.
    bull_value, bear_value : float
        Value produced by the numba backend at the completion bar.
    bull_idx, bear_idx : int, optional
        Index of the flagged bar within the full sequence (flat prefix
        included); default: last bar.
    talib_values : tuple
        Values the talib backend may produce (normalized convention).
    extra : dict, optional
        Extra kwargs forwarded to the universal function in all tests.
    flat : tuple
        Neutral candle used for prefixes and the no-pattern test.
    prefix_n : int
        Number of flat candles prepended before the pattern.

    """
    extra = extra or {}
    tests = {}

    def _run(candles, use_talib, **kw):
        o, h, low_, c = ohlc(candles)
        return fn(o, h, low_, c, use_talib=use_talib, **{**extra, **kw})

    def _idx(candles, idx):
        return len(candles) - 1 if idx is None else idx

    def _full(candles):
        return with_prefix(candles, n=prefix_n, flat=flat)

    # ---- bullish detection (numba backend) -------------------------------
    if bull is not None:
        def test_bullish_detected():
            full = _full(bull)
            i = _idx(full, bull_idx)
            r = _run(full, use_talib=False)
            assert r.dtype == np.float64
            assert r[i] == bull_value
            assert_array_equal(r[:4], np.zeros(4))
        tests[f'test_{name}_bullish_detected'] = test_bullish_detected

        def test_flat_input_no_pattern():
            candles = [flat] * 12
            r = _run(candles, use_talib=False)
            assert_array_equal(r, np.zeros(len(candles)))
        tests[f'test_{name}_flat_input_no_pattern'] = \
            test_flat_input_no_pattern

        def test_talib_backend_binary():
            if not talib_available:
                pytest.skip('TA-Lib not installed')
            full = _full(bull)
            r = _run(full, use_talib=True)
            assert np.all(np.isin(np.unique(r), list(talib_values)))
        tests[f'test_{name}_talib_backend_binary'] = test_talib_backend_binary

        def test_readonly_polars_series_input():
            full = _full(bull)
            o, h, low_, c = ohlc(full)
            r_series = fn(
                pl.Series(o), pl.Series(h), pl.Series(low_), pl.Series(c),
                use_talib=False, **extra
            )
            r_arr = _run(full, use_talib=False)
            assert_array_equal(r_series, r_arr)
        tests[f'test_{name}_readonly_polars_input'] = \
            test_readonly_polars_series_input

        def test_offset_and_fillna():
            full = _full(bull)
            base = _run(full, use_talib=False)
            shifted = _run(full, use_talib=False, offset=2, fillna=-7.0)
            assert_allclose(shifted[:2], -7.0, rtol=0, atol=0)
            assert_allclose(shifted[2:], base[:-2], rtol=0, atol=0)
        tests[f'test_{name}_offset_and_fillna'] = test_offset_and_fillna

    # ---- bearish detection (numba backend) -------------------------------
    if bear is not None:
        def test_bearish_detected():
            full = _full(bear)
            i = _idx(full, bear_idx)
            r = _run(full, use_talib=False)
            assert r[i] == bear_value
            assert_array_equal(r[:4], np.zeros(4))
        tests[f'test_{name}_bearish_detected'] = test_bearish_detected

    # ---- polars wrapper ---------------------------------------------------
    if polars_fn is not None and output_col is not None and bull is not None:
        def test_polars_wrapper():
            full = _full(bull)
            o, h, low_, c = ohlc(full)
            df = pl.DataFrame({
                'open': o, 'high': h, 'low': low_, 'close': c
            })
            out = polars_fn(df)
            assert output_col in out.columns
            assert len(out) == len(df)
            col = out[output_col].to_numpy()
            # the wrapper uses the module default backend
            expected = _run(full, use_talib=True)
            assert_array_equal(col, expected)
            assert np.all(np.isin(np.unique(col), list(talib_values)))
        tests[f'test_{name}_polars_wrapper'] = test_polars_wrapper

    return tests
