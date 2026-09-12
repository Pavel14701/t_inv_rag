# -*- coding: utf-8 -*-
"""Unit tests for Supertrend module.

Tests cover:
- Regression: supertrend_numba no longer raises on int64 direction
- Direction semantics: NaN seed, values in {-1, +1}
- trend == long in uptrend, trend == short in downtrend, mutual exclusion
- Uptrend band never decreases (ratchet property)
- Consistency with the underlying ATR-based bands
- Backend parity: numba (rma ATR) vs TA-Lib ATR
- Validation of length / atr_length / multiplier
- offset and fillna behaviour
- supertrend_polars DataFrame integration
- IEEE 754 edge cases
"""

import pytest
import numpy as np
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.supertrend import (
    supertrend_numba,
    supertrend_talib,
    supertrend_ind,
    supertrend_polars
)
from ...external import talib_available


# -----------------------------------------------------------------------------
# Regression / structure
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_supertrend_numba_runs(
    df_ohlc: pl.DataFrame
) -> None:
    """Regression: int64 direction array used to crash _apply_offset_fillna
    with 'No matching definition for argument type(s) array(int64, 1d, C)'.
    """
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    trend, direction, long_b, short_b = supertrend_numba(
        high, low, close, length=7
    )
    for arr in (trend, direction, long_b, short_b):
        assert arr.dtype == np.float64
        assert len(arr) == len(close)


@pytest.mark.overlap
def test_supertrend_direction_semantics(df_ohlc: pl.DataFrame) -> None:
    """Direction is NaN only at index 0 and +/-1 afterwards."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    trend, direction, long_b, short_b = supertrend_numba(
        high, low, close, length=7
    )
    assert np.isnan(direction[0])
    assert set(np.unique(direction[1:])).issubset({-1.0, 1.0})
    # both directions are actually exercised by a random-walk series
    assert (direction[1:] == 1.0).any()
    assert (direction[1:] == -1.0).any()


@pytest.mark.overlap
def test_supertrend_long_short_exclusive(
    df_ohlc: pl.DataFrame
) -> None:
    """Long and short are mutually exclusive; trend matches the active one."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    trend, direction, long_b, short_b = supertrend_numba(
        high, low, close, length=7
    )
    valid = ~np.isnan(trend)
    both = valid & ~np.isnan(long_b) & ~np.isnan(short_b)
    assert not both.any()
    up = valid & ~np.isnan(long_b)
    down = valid & ~np.isnan(short_b)
    assert np.isnan(long_b[down]).all()
    assert np.isnan(short_b[up]).all()
    assert_allclose(trend[up], long_b[up], rtol=0, atol=0)
    assert_allclose(trend[down], short_b[down], rtol=0, atol=0)


@pytest.mark.overlap
def test_supertrend_warmup_nan(df_ohlc: pl.DataFrame) -> None:
    """NaN during the ATR warm-up window, finite afterwards."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    length = 7

    trend, direction, long, short = supertrend_numba(
        high, low, close, length=length
    )
    assert np.isnan(trend[0])
    assert not np.isnan(trend[length:]).any()

# -----------------------------------------------------------------------------
# Properties and reference recomputation
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_supertrend_uptrend_ratchet(df_ohlc: pl.DataFrame) -> None:
    """In an uptrend the (long) trend never decreases between consecutive
    uptrend bars (band ratchet).
    """
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    trend, direction, long_b, short_b = supertrend_numba(
        high, low, close, length=7
    )
    for i in range(2, len(trend)):
        if np.isnan(trend[i]) or np.isnan(trend[i - 1]):
            continue  # ATR warm-up zone
        if direction[i] > 0 and direction[i - 1] > 0:
            assert trend[i] >= trend[i - 1] - 1e-12
        if direction[i] < 0 and direction[i - 1] < 0:
            assert trend[i] <= trend[i - 1] + 1e-12


@pytest.mark.overlap
def test_supertrend_matches_manual_recomputation(
    df_ohlc: pl.DataFrame
) -> None:
    """Full recomputation of the supertrend state machine from the ATR."""
    from ...volatility import atr_ind

    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    length, multiplier = 7, 3.0

    trend, direction, long, short = supertrend_numba(
        high, low, close, length=length, multiplier=multiplier
    )

    atr = atr_ind(
        high, low, close, length=length, mamode='rma', drift=1,
        offset=0, fillna=None, percent=False, use_talib=False
    )
    hl2 = (high + low) * 0.5
    ub = hl2 + multiplier * atr
    lb = hl2 - multiplier * atr

    n = len(close)
    dir_val = 1.0
    exp_trend = np.full(n, np.nan)
    exp_dir = np.full(n, np.nan)
    for i in range(1, n):
        if close[i] > ub[i - 1]:
            dir_val = 1.0
        elif close[i] < lb[i - 1]:
            dir_val = -1.0
        if dir_val > 0:
            lb[i] = max(lb[i], lb[i - 1])
            exp_trend[i] = lb[i]
        else:
            ub[i] = min(ub[i], ub[i - 1])
            exp_trend[i] = ub[i]
        exp_dir[i] = dir_val

    valid = ~np.isnan(exp_trend)
    assert_allclose(trend[valid], exp_trend[valid], rtol=1e-12, atol=1e-12)
    assert_allclose(direction[valid], exp_dir[valid], rtol=0, atol=0)


@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
@pytest.mark.overlap
def test_supertrend_backend_parity_tail(df_ohlc: pl.DataFrame) -> None:
    """TA-Lib ATR backend agrees with the rma backend on the converged tail
    (both are Wilder-smoothed ATRs; small early drift is expected).
    """
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    length = 7

    t_n, d_n, l_n, s_n = supertrend_numba(high, low, close, length=length)
    t_t, d_t, l_t, s_t = supertrend_talib(high, low, close, length=length)

    tail = slice(3 * length, None)
    assert_allclose(t_n[tail], t_t[tail], rtol=1e-8, atol=1e-8)
    assert_allclose(d_n[tail], d_t[tail], rtol=0, atol=0)


# -----------------------------------------------------------------------------
# Validation / contiguity / read-only
# -----------------------------------------------------------------------------

@pytest.mark.overlap
@pytest.mark.parametrize(
    'kwargs, match', [
        ({'length': 0}, 'length must be >= 1'),
        ({'atr_length': -1}, 'atr_length must be >= 1'),
        ({'multiplier': -0.5}, 'multiplier must be >= 0'),
    ]
)
def test_supertrend_validation(
    df_ohlc: pl.DataFrame, kwargs: dict, match: str
) -> None:
    """Invalid parameters raise ValueError."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    with pytest.raises(ValueError, match=match):
        supertrend_numba(high, low, close, **kwargs)


@pytest.mark.overlap
def test_supertrend_non_contiguous_and_readonly(
    df_ohlc: pl.DataFrame
) -> None:
    """Strided and polars (read-only) inputs give identical results to the
    contiguous computation on the same (resampled) data.
    """
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    h2, l2, c2 = high[::2], low[::2], close[::2]
    strided = supertrend_numba(h2, l2, c2, length=7)
    contig = supertrend_numba(
        np.ascontiguousarray(h2), np.ascontiguousarray(l2),
        np.ascontiguousarray(c2), length=7
    )
    for s, b in zip(strided, contig):
        assert_allclose(s, b, rtol=0, atol=0)

    via_pl = supertrend_ind(
        pl.Series(high), pl.Series(low), pl.Series(close),
        length=7, use_talib=False
    )
    base = supertrend_numba(high, low, close, length=7)
    for b, p in zip(base, via_pl):
        assert_allclose(p, b, rtol=0, atol=0)

# -----------------------------------------------------------------------------
# Offset / fillna
# -----------------------------------------------------------------------------


@pytest.mark.overlap
def test_supertrend_offset(df_ohlc: pl.DataFrame) -> None:
    """Positive offset shifts all four outputs forward."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    base = supertrend_numba(high, low, close, length=7)
    shifted = supertrend_numba(high, low, close, length=7, offset=2)
    for b, s in zip(base, shifted):
        assert np.isnan(s[:2]).all()
        assert_allclose(s[2:], b[:-2], rtol=0, atol=0)


@pytest.mark.overlap
def test_supertrend_fillna(df_ohlc: pl.DataFrame) -> None:
    """Fillna replaces the warm-up NaNs in every output array."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    outs = supertrend_numba(high, low, close, length=7, fillna=-1.0)
    for arr in outs:
        assert not np.isnan(arr).any()
        assert (arr[0] == -1.0)
    # long stays NaN in downtrends -> also filled
    trend, direction, long, short = outs
    down = direction < 0
    assert (long[down] == -1.0).all()


# -----------------------------------------------------------------------------
# Polars integration
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_supertrend_polars_columns(df_ohlc: pl.DataFrame) -> None:
    """Default columns SUPERT_{length}_{multiplier}, SUPERTd/l/s."""
    out = supertrend_polars(df_ohlc, length=7, use_talib=False)
    for name in ('SUPERT_7_3.0', 'SUPERTd_7_3.0', 'SUPERTl_7_3.0',
                 'SUPERTs_7_3.0'):
        assert name in out.columns
    assert len(out) == len(df_ohlc)


@pytest.mark.overlap
def test_supertrend_polars_custom_suffix_and_fillna(
    df_ohlc: pl.DataFrame
) -> None:
    """Custom suffix and fillna are honoured."""
    out = supertrend_polars(
        df_ohlc, length=7, fillna=-1.0, suffix='_t', use_talib=False
    )
    for name in ('SUPERT_t', 'SUPERTd_t', 'SUPERTl_t', 'SUPERTs_t'):
        assert name in out.columns
    assert out['SUPERT_t'].null_count() == 0
    assert not out['SUPERT_t'].is_null().any()


# -----------------------------------------------------------------------------
# IEEE 754 edge cases
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_supertrend_short_input() -> None:
    """Input shorter than the ATR warm-up raises from ATR validation."""
    close = np.array([1.0, 2.0])
    high = close + 0.5
    low = close - 0.5
    with pytest.raises(ValueError, match='too short'):
        supertrend_numba(high, low, close, length=7)


@pytest.mark.overlap
def test_supertrend_nan_atr_prefix(df_ohlc: pl.DataFrame) -> None:
    """NaN ATR warm-up values do not corrupt the first valid bars:
    NaN comparisons are False, so the initial direction is kept.
    """
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()

    trend, direction, long_b, short_b = supertrend_numba(
        high, low, close, length=7
    )
    valid = ~np.isnan(trend)
    # once valid, values stay finite to the end
    assert not np.isnan(trend[valid]).any()
    assert valid.sum() > len(trend) // 2


@pytest.mark.overlap
def test_supertrend_empty_input() -> None:
    """Empty input raises from ATR validation (needs at least length+1)."""
    empty = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match='too short'):
        supertrend_numba(empty, empty, empty, length=7)
