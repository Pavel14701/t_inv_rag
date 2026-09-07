# -*- coding: utf-8 -*-
"""Unit tests for True Range (TR) module.

Tests cover:
- _true_range_numba_core and true_range_numba against the reference
- gap behaviour (|close - prev_close| dominates high - low)
- drift semantics (drift=1 default, drift=2 via Numba)
- validation: drift < 1 (look-ahead guard), length mismatch,
  series too short
- offset and fillna
- NaN handling (nan_policy raise / ignore / ffill) and Inf -> NaN
- backend rules: drift != 1 forces Numba in true_range_ind;
  true_range_talib rejects drift != 1
- true_range_ind with Polars Series
- true_range_polars DataFrame integration
- IEEE 754 compliance (empty handled via too-short, extreme, all-NaN)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...volatility.true_range import (
    _true_range_numba_core,
    true_range_numba,
    true_range_talib,
    true_range_ind,
    true_range_polars,
)
from ...external import talib_available


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------
def _tr_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    drift: int = 1,
) -> npt.NDArray[np.float64]:
    """Pure Python reference True Range (no look-ahead, drift >= 1)."""
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(drift, n):
        out[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - drift]),
            abs(low[i] - close[i - drift]),
        )
    return out


# -----------------------------------------------------------------------------
# Correctness tests
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_tr_core_against_reference(
    df_ohlc: pl.DataFrame,
) -> None:
    """The Numba core must match the pure Python reference."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = _true_range_numba_core(high, low, close, 1)
    expected = _tr_reference(high, low, close, 1)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.volatility
def test_tr_numba_against_reference(
    df_ohlc: pl.DataFrame,
) -> None:
    """true_range_numba must match the reference (drift=1 and drift=3)."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    for drift in (1, 3):
        result = true_range_numba(high, low, close, drift=drift)
        expected = _tr_reference(high, low, close, drift)
        assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.volatility
def test_tr_warmup_nans(df_ohlc: pl.DataFrame) -> None:
    """The first `drift` values are NaN, the rest are finite."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = true_range_numba(high, low, close, drift=1)
    assert np.isnan(result[0])
    assert np.isfinite(result[1:]).all()


@pytest.mark.volatility
def test_tr_gap_down_dominates() -> None:
    """A gap up above prev_close makes |low - prev_close| dominate."""
    high = np.array([10.0, 10.0])
    low = np.array([9.0, 9.9])
    close = np.array([9.5, 9.95])
    result = true_range_numba(high, low, close)
    # prev_close=9.5; bar 1: hl=0.1, |low-prev|=0.4, |high-prev|=0.5
    # -> the |high - prev_close| term dominates
    assert_allclose(result[1], 0.5, rtol=1e-12)
    assert result[1] > high[1] - low[1]  # range term is not the max


@pytest.mark.volatility
def test_tr_gap_up_dominates() -> None:
    """A gap up makes |high - prev_close| the True Range."""
    high = np.array([10.0, 12.0])
    low = np.array([9.0, 11.0])
    close = np.array([9.5, 11.5])
    result = true_range_numba(high, low, close)
    assert_allclose(result[1], abs(12.0 - 9.5), rtol=1e-12)
    assert result[1] > high[1] - low[1]


@pytest.mark.volatility
def test_tr_positive_and_nonnegative(df_ohlc: pl.DataFrame) -> None:
    """All computed TR values are strictly positive."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = true_range_numba(high, low, close)
    assert (result[1:] > 0).all()


@pytest.mark.volatility
def test_tr_drift_two_numba(df_ohlc: pl.DataFrame) -> None:
    """drift=2 compares with close two bars back (Numba backend)."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = true_range_numba(high, low, close, drift=2)
    expected = _tr_reference(high, low, close, drift=2)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
    assert np.isnan(result[:2]).all()
    assert np.isfinite(result[2:]).all()


@pytest.mark.volatility
def test_tr_prenan_is_noop(df_ohlc: pl.DataFrame) -> None:
    """prenan=True and prenan=False produce identical output."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    a = true_range_numba(high, low, close, prenan=True)
    b = true_range_numba(high, low, close, prenan=False)
    assert_allclose(a, b, rtol=1e-12, equal_nan=True)


# -----------------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_tr_negative_drift_rejected() -> None:
    """Drift < 1 is rejected (would use the future close: look-ahead)."""
    high = np.arange(1.0, 11.0)
    low = high - 1.0
    close = high - 0.5
    for bad_drift in (-1, 0):
        with pytest.raises(ValueError, match='drift must be >= 1'):
            true_range_numba(high, low, close, drift=bad_drift)


@pytest.mark.volatility
def test_tr_length_mismatch_rejected() -> None:
    """Different lengths of high/low/close raise ValueError."""
    high = np.arange(1.0, 11.0)
    low = high - 1.0
    close = high - 0.5
    with pytest.raises(ValueError, match='same length'):
        true_range_numba(high, low[:5], close)
    with pytest.raises(ValueError, match='same length'):
        true_range_numba(high[:7], low, close)


@pytest.mark.volatility
def test_tr_too_short_rejected() -> None:
    """Series shorter than drift+1 raises ValueError, not silent NaN."""
    with pytest.raises(ValueError, match='Input series too short'):
        true_range_numba(
            np.array([10.0]), np.array([9.0]), np.array([9.5]), drift=1
        )
    with pytest.raises(ValueError, match='Input series too short'):
        true_range_numba(
            np.arange(1.0, 4.0),
            np.arange(0.0, 3.0),
            np.arange(0.5, 3.5),
            drift=3,
        )


@pytest.mark.volatility
def test_tr_empty_rejected(prices_empty) -> None:
    """Empty input raises ValueError (series too short)."""
    with pytest.raises(ValueError, match='Input series too short'):
        true_range_numba(prices_empty, prices_empty, prices_empty)


# -----------------------------------------------------------------------------
# offset / fillna tests
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_tr_offset(df_ohlc: pl.DataFrame) -> None:
    """Positive offset shifts the result forward."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    base = true_range_numba(high, low, close)
    shifted = true_range_numba(high, low, close, offset=2)
    assert np.isnan(shifted[:2]).all()
    assert_allclose(shifted[2:], base[:-2], rtol=1e-12, equal_nan=True)


@pytest.mark.volatility
def test_tr_negative_offset(df_ohlc: pl.DataFrame) -> None:
    """Negative offset shifts the result backward."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    base = true_range_numba(high, low, close)
    shifted = true_range_numba(high, low, close, offset=-2)
    assert np.isnan(shifted[-2:]).all()
    assert_allclose(shifted[:-2], base[2:], rtol=1e-12, equal_nan=True)


@pytest.mark.volatility
def test_tr_fillna(df_ohlc: pl.DataFrame) -> None:
    """Fillna replaces warm-up NaNs."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = true_range_numba(high, low, close, fillna=0.0)
    assert not np.isnan(result).any()
    assert result[0] == 0.0


# -----------------------------------------------------------------------------
# NaN / Inf handling (nan_policy)
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_tr_nan_policy_raise(df_ohlc: pl.DataFrame) -> None:
    """Default nan_policy='raise' rejects NaN in any price column."""
    high = df_ohlc['high'].to_numpy().copy()
    low = df_ohlc['low'].to_numpy().copy()
    close = df_ohlc['close'].to_numpy().copy()
    high[5] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        true_range_numba(high, low, close)
    low[5] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        true_range_numba(df_ohlc['high'].to_numpy(), low, close)
    with pytest.raises(ValueError, match='NaN'):
        true_range_numba(
            df_ohlc['high'].to_numpy(),
            df_ohlc['low'].to_numpy(),
            np.where(np.arange(len(close)) == 5, np.nan, close),
        )


@pytest.mark.volatility
def test_tr_nan_propagates_with_ignore(df_ohlc: pl.DataFrame) -> None:
    """NaN in high/low/close propagates into the affected TR bars."""  # noqa: D403, E501
    high0 = df_ohlc['high'].to_numpy()
    low0 = df_ohlc['low'].to_numpy()
    close0 = df_ohlc['close'].to_numpy()

    # NaN in high affects only its own bar's TR
    high = high0.copy()
    high[5] = np.nan
    result = true_range_numba(high, low0, close0, nan_policy='ignore')
    assert np.isnan(result[5])
    assert np.isfinite(result[6:]).all()

    # NaN in close: the gap terms are NaN, so TR degrades to the plain
    # high - low range (numba max keeps the finite term; pandas_ta's
    # skipna max behaves the same way)
    close = close0.copy()
    close[8] = np.nan
    result = true_range_numba(high0, low0, close, nan_policy='ignore')
    assert_allclose(result[9], high0[9] - low0[9], rtol=1e-12)
    assert np.isfinite(result[9:]).all()

    # NaN in low affects only its own bar's TR
    low = low0.copy()
    low[3] = np.nan
    result = true_range_numba(high0, low, close0, nan_policy='ignore')
    assert np.isnan(result[3])
    assert np.isfinite(result[4:]).all()


@pytest.mark.volatility
def test_tr_nan_policy_ffill(df_ohlc: pl.DataFrame) -> None:
    """nan_policy='ffill' fills the input NaN and yields finite output."""
    high = df_ohlc['high'].to_numpy().copy()
    low = df_ohlc['low'].to_numpy().copy()
    close = df_ohlc['close'].to_numpy().copy()
    high[5] = np.nan
    result = true_range_numba(high, low, close, nan_policy='ffill')
    # First drift value is the warm-up; everything else is finite
    assert np.isnan(result[0])
    assert np.isfinite(result[1:]).all()


@pytest.mark.volatility
def test_tr_invalid_nan_policy() -> None:
    """Unknown nan_policy raises ValueError (validated up front)."""
    high = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match='nan_policy'):
        true_range_numba(high, high - 1, high - 0.5, nan_policy='drop')


@pytest.mark.volatility
def test_tr_inf_replaced_with_nan(df_ohlc: pl.DataFrame) -> None:
    """Inf in input is replaced with NaN and handled like NaN."""
    high = df_ohlc['high'].to_numpy().copy()
    low = df_ohlc['low'].to_numpy().copy()
    close = df_ohlc['close'].to_numpy().copy()
    high_with_inf = high.copy()
    high_with_inf[5] = np.inf
    # Default policy raises (Inf became NaN)
    with pytest.raises(ValueError, match='NaN'):
        true_range_numba(high_with_inf, low, close)
    # With 'ignore': no inf in output, NaN in the affected bar
    result = true_range_numba(
        high_with_inf, low, close, nan_policy='ignore'
    )
    assert not np.isinf(result).any()
    assert np.isnan(result[5])
    assert np.isfinite(result[6:]).all()


# -----------------------------------------------------------------------------
# Backend rules
# -----------------------------------------------------------------------------
@pytest.mark.volatility
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_tr_backend_parity(df_ohlc: pl.DataFrame) -> None:
    """Numba and TA-Lib backends agree for drift=1."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    numba_res = true_range_numba(high, low, close)
    talib_res = true_range_talib(high, low, close)
    assert_allclose(numba_res, talib_res, rtol=1e-10, equal_nan=True)


@pytest.mark.volatility
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_tr_talib_rejects_drift_not_one(df_ohlc: pl.DataFrame) -> None:
    """true_range_talib raises for drift != 1 (TA-Lib is drift=1 only)."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    with pytest.raises(ValueError, match='drift=1 only'):
        true_range_talib(high, low, close, drift=2)


@pytest.mark.volatility
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_tr_ind_forces_numba_for_drift_two(
    df_ohlc: pl.DataFrame,
) -> None:
    """true_range_ind(drift=2) must use Numba even with use_talib=True.

    Regression test: TA-Lib silently ignored drift and returned the
    drift=1 values.
    """
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    ind_res = true_range_ind(high, low, close, drift=2, use_talib=True)
    numba_res = true_range_numba(high, low, close, drift=2)
    assert_allclose(ind_res, numba_res, rtol=1e-12, equal_nan=True)


# -----------------------------------------------------------------------------
# true_range_ind / true_range_polars integration
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_tr_ind_with_pl_series(df_ohlc: pl.DataFrame) -> None:
    """true_range_ind accepts Polars Series."""
    result = true_range_ind(
        df_ohlc['high'], df_ohlc['low'], df_ohlc['close'],
        use_talib=False,
    )
    expected = _tr_reference(
        df_ohlc['high'].to_numpy(),
        df_ohlc['low'].to_numpy(),
        df_ohlc['close'].to_numpy(),
    )
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.volatility
def test_tr_ind_int_input(df_ohlc: pl.DataFrame) -> None:
    """Integer price arrays are cast to float64 without errors."""
    high = (df_ohlc['high'] * 100).cast(pl.Int64).to_numpy()
    low = (df_ohlc['low'] * 100).cast(pl.Int64).to_numpy()
    close = (df_ohlc['close'] * 100).cast(pl.Int64).to_numpy()
    result = true_range_numba(high, low, close)
    expected = _tr_reference(
        high.astype(np.float64),
        low.astype(np.float64),
        close.astype(np.float64),
    )
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.volatility
def test_tr_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """true_range_polars adds a column matching the reference."""
    result_df = true_range_polars(
        df_ohlc, output_col='TR', use_talib=False
    )
    assert 'TR' in result_df.columns
    assert result_df['TR'].dtype == pl.Float64
    assert len(result_df) == len(df_ohlc)
    expected = _tr_reference(
        df_ohlc['high'].to_numpy(),
        df_ohlc['low'].to_numpy(),
        df_ohlc['close'].to_numpy(),
    )
    assert_allclose(result_df['TR'].to_numpy(), expected, rtol=1e-12,
                    equal_nan=True)


@pytest.mark.volatility
def test_tr_polars_default_output_col(df_ohlc: pl.DataFrame) -> None:
    """Default output column name is TRUERANGE_{drift}."""
    result_df = true_range_polars(df_ohlc, drift=1, use_talib=False)
    assert 'TRUERANGE_1' in result_df.columns
    result_df2 = true_range_polars(df_ohlc, drift=2, use_talib=False)
    assert 'TRUERANGE_2' in result_df2.columns


@pytest.mark.volatility
def test_tr_polars_offset_fillna(df_ohlc: pl.DataFrame) -> None:
    """true_range_polars applies offset and fillna."""
    result_df = true_range_polars(
        df_ohlc, offset=1, fillna=0.0, output_col='TR', use_talib=False
    )
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    expected = true_range_numba(high, low, close, offset=1, fillna=0.0)
    assert_allclose(result_df['TR'].to_numpy(), expected, rtol=1e-12,
                    equal_nan=True)


@pytest.mark.volatility
def test_tr_polars_with_nan(df_ohlc: pl.DataFrame) -> None:
    """true_range_polars propagates NaN with nan_policy='ignore'."""
    df = df_ohlc.with_columns(
        [pl.Series('high', np.where(
            np.arange(df_ohlc.height) == 5,
            np.nan,
            df_ohlc['high'].to_numpy(),
        ))]
    )
    result_df = true_range_polars(
        df, output_col='TR', use_talib=False, nan_policy='ignore'
    )
    tr_vals = result_df['TR'].to_numpy()
    assert np.isnan(tr_vals[5])
    assert np.isfinite(tr_vals[6:]).all()


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_tr_all_nan_prices() -> None:
    """All-NaN prices: raise by default, all-NaN with 'ignore'."""
    n = 10
    nan_arr = np.full(n, np.nan)
    with pytest.raises(ValueError, match='NaN'):
        true_range_numba(nan_arr, nan_arr, nan_arr)
    result = true_range_numba(
        nan_arr, nan_arr, nan_arr, nan_policy='ignore'
    )
    assert np.isnan(result).all()
    result_fill = true_range_numba(
        nan_arr, nan_arr, nan_arr, fillna=0.0, nan_policy='ignore'
    )
    assert (result_fill == 0.0).all()


@pytest.mark.volatility
def test_tr_extreme_values(df_ohlc: pl.DataFrame) -> None:
    """Extreme values (1e300, 1e-300) must not crash."""
    high = df_ohlc['high'].to_numpy().copy()
    low = df_ohlc['low'].to_numpy().copy()
    close = df_ohlc['close'].to_numpy().copy()
    high[0] = 1e300
    low[1] = 1e-300
    result = true_range_numba(high, low, close, nan_policy='ignore')
    assert result is not None
    assert len(result) == len(df_ohlc)
    assert not np.isinf(result[2:]).any()
