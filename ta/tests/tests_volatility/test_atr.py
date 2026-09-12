# -*- coding: utf-8 -*-
"""Unit tests for Average True Range (ATR) module.

Tests cover:
- atr_numba against an independent RMA/SMA/EMA-of-TR reference
- talib parity for the default Wilder mode (drift=1)
- warm-up semantics (drift + length - 1 leading NaNs)
- mamode routing ('rma', 'sma', 'ema'), percent mode, trim
- backend rules: mamode != 'rma' or drift != 1 force Numba in atr_ind
  (regression: TA-Lib silently ignored these parameters)
- validation: length/drift, length mismatch, series too short,
  invalid mamode, invalid nan_policy
- NaN handling (nan_policy raise / ignore / ffill) and Inf -> NaN
- atr_ind with Polars Series and integer inputs
- atr_polars DataFrame integration
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose

from ...volatility.atr import atr_numba, atr_talib, atr_ind, atr_polars
from ...external import talib_available


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------
def _tr_series(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    drift: int,
) -> npt.NDArray[np.float64]:
    """Pure Python True Range (same formula as the TR module)."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(drift, n):
        out[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - drift]),
            abs(low[i] - close[i - drift]),
        )
    return out


def _atr_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    length: int,
    mamode: str = 'rma',
    drift: int = 1,
) -> npt.NDArray[np.float64]:
    """Pure Python ATR: MA over the TR valid part, warm-up preserved."""
    n = len(high)
    tr = _tr_series(high, low, close, drift)
    valid = tr[drift:]
    out = np.full(n, np.nan)
    if mamode == 'sma':
        for i in range(length - 1, len(valid)):
            out[i + drift] = valid[i - length + 1: i + 1].mean()
    elif mamode == 'ema':
        k = 2.0 / (length + 1)
        prev = valid[:length].mean()
        out[length - 1 + drift] = prev
        for i in range(length, len(valid)):
            prev = (valid[i] - prev) * k + prev
            out[i + drift] = prev
    else:  # 'rma' (Wilder)
        prev = valid[:length].mean()
        out[length - 1 + drift] = prev
        for i in range(length, len(valid)):
            prev = (prev * (length - 1) + valid[i]) / length
            out[i + drift] = prev
    return out


# -----------------------------------------------------------------------------
# Correctness tests
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_atr_numba_rma_against_reference(df_ohlc: pl.DataFrame) -> None:
    """atr_numba (default Wilder RMA) matches the pure Python reference."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = atr_numba(high, low, close, length=14)
    expected = _atr_reference(high, low, close, length=14, mamode='rma')
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.volatility
def test_atr_numba_sma_and_ema_against_reference(
    df_ohlc: pl.DataFrame,
) -> None:
    """atr_numba SMA/EMA modes match the pure Python references."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    for mode in ('sma', 'ema'):
        result = atr_numba(high, low, close, length=10, mamode=mode)
        expected = _atr_reference(
            high, low, close, length=10, mamode=mode
        )
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.volatility
def test_atr_numba_drift_two(df_ohlc: pl.DataFrame) -> None:
    """drift=2: TR warm-up is 2 bars, ATR warm-up is drift+length-1."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = atr_numba(high, low, close, length=10, drift=2)
    expected = _atr_reference(high, low, close, length=10, drift=2)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)
    assert np.isnan(result[:11]).all()  # drift + length - 1 = 11
    assert np.isfinite(result[11:]).all()


@pytest.mark.volatility
def test_atr_warmup_nans(df_ohlc: pl.DataFrame) -> None:
    """Default warm-up is drift + length - 1 = 14 leading NaNs."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = atr_numba(high, low, close, length=14)
    assert np.isnan(result[:14]).all()
    assert np.isfinite(result[14:]).all()
    assert len(result) == len(close)


@pytest.mark.volatility
def test_atr_percent_mode(df_ohlc: pl.DataFrame) -> None:
    """percent=True returns ATR as a percentage of close."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    base = atr_numba(high, low, close, length=14)
    pct = atr_numba(high, low, close, length=14, percent=True)
    assert_allclose(pct[14:], base[14:] * 100.0 / close[14:], rtol=1e-10)


@pytest.mark.volatility
def test_atr_trim(df_ohlc: pl.DataFrame) -> None:
    """Trim removes the full warm-up (drift + length - 1 bars)."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    result = atr_numba(high, low, close, length=14, trim=True)
    assert len(result) == len(high) - 14
    assert np.isfinite(result).all()


# -----------------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_atr_invalid_length_and_drift(df_ohlc: pl.DataFrame) -> None:
    """Length < 1 and drift < 1 raise ValueError."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    with pytest.raises(ValueError, match='length must be >= 1'):
        atr_numba(high, low, close, length=0)
    for bad_drift in (0, -1):
        with pytest.raises(ValueError, match='drift must be >= 1'):
            atr_numba(high, low, close, drift=bad_drift)


@pytest.mark.volatility
def test_atr_length_mismatch(df_ohlc: pl.DataFrame) -> None:
    """Different lengths of high/low/close raise ValueError."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    with pytest.raises(ValueError, match='same length'):
        atr_numba(high, low[:5], close)
    with pytest.raises(ValueError, match='same length'):
        atr_talib(high[:7], low, close)


@pytest.mark.volatility
def test_atr_too_short(df_ohlc: pl.DataFrame) -> None:
    """Series shorter than drift + length raises ValueError."""
    high = df_ohlc['high'].to_numpy()[:10]
    low = df_ohlc['low'].to_numpy()[:10]
    close = df_ohlc['close'].to_numpy()[:10]
    with pytest.raises(ValueError, match='Input series too short'):
        atr_numba(high, low, close, length=14)
    with pytest.raises(ValueError, match='Input series too short'):
        atr_numba(high, low, close, length=5, drift=10)


@pytest.mark.volatility
def test_atr_invalid_mamode(df_ohlc: pl.DataFrame) -> None:
    """Unsupported mamode raises ValueError."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    with pytest.raises(ValueError, match='Unsupported mamode'):
        atr_numba(high, low, close, mamode='wma')


@pytest.mark.volatility
def test_atr_invalid_nan_policy(df_ohlc: pl.DataFrame) -> None:
    """Unknown nan_policy raises ValueError (validated up front)."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    with pytest.raises(ValueError, match='nan_policy'):
        atr_numba(high, low, close, nan_policy='drop')


# -----------------------------------------------------------------------------
# NaN / Inf handling (nan_policy)
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_atr_nan_policy_raise(df_ohlc: pl.DataFrame) -> None:
    """Default nan_policy='raise' rejects NaN in any price column."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    for arr_idx in range(3):
        prices = [high.copy(), low.copy(), close.copy()]
        prices[arr_idx][5] = np.nan
        with pytest.raises(ValueError, match='NaN'):
            atr_numba(prices[0], prices[1], prices[2])


@pytest.mark.volatility
def test_atr_nan_propagates_with_ignore(df_ohlc: pl.DataFrame) -> None:
    """NaN in prices with 'ignore' poisons the recursive ATR onwards.

    RMA/EMA are recursive: a single NaN in the input permanently
    propagates into all later values (documented behaviour).
    """
    high = df_ohlc['high'].to_numpy().copy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    high[20] = np.nan
    result = atr_numba(high, low, close, length=5, nan_policy='ignore')
    # Warm-up finite, from the NaN bar onwards the recursive RMA is
    # permanently poisoned (NaN propagates through the recursion).
    assert np.isnan(result[20:]).all()
    assert np.isfinite(result[5:20]).all()


@pytest.mark.volatility
def test_atr_nan_policy_ffill(df_ohlc: pl.DataFrame) -> None:
    """nan_policy='ffill' fills the input NaN and yields finite output."""
    high = df_ohlc['high'].to_numpy().copy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    high[20] = np.nan
    result = atr_numba(high, low, close, length=5, nan_policy='ffill')
    # Warm-up (drift + length - 1 = 5) NaN, everything else finite
    assert np.isnan(result[:5]).all()
    assert np.isfinite(result[5:]).all()


@pytest.mark.volatility
def test_atr_inf_replaced_with_nan(df_ohlc: pl.DataFrame) -> None:
    """Inf in input is replaced with NaN and handled like NaN."""
    high = df_ohlc['high'].to_numpy().copy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    high_with_inf = high.copy()
    high_with_inf[20] = np.inf
    # Default policy raises (Inf became NaN)
    with pytest.raises(ValueError, match='NaN'):
        atr_numba(high_with_inf, low, close)
    # With 'ffill': no inf anywhere, finite output after warm-up
    result = atr_numba(
        high_with_inf, low, close, length=5, nan_policy='ffill'
    )
    assert not np.isinf(result).any()
    assert np.isfinite(result[5:]).all()


# -----------------------------------------------------------------------------
# Backend rules
# -----------------------------------------------------------------------------
@pytest.mark.volatility
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_atr_backend_parity_wilder(df_ohlc: pl.DataFrame) -> None:
    """Numba and TA-Lib agree for the default Wilder mode (drift=1)."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    numba_res = atr_numba(high, low, close, length=14)
    talib_res = atr_talib(high, low, close, length=14)
    assert_allclose(numba_res, talib_res, rtol=1e-8, equal_nan=True)


@pytest.mark.volatility
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_atr_ind_forces_numba_for_sma_mode(df_ohlc: pl.DataFrame) -> None:
    """atr_ind(mamode='sma') must use Numba even with use_talib=True.

    Regression test: TA-Lib ATR is hard-wired to Wilder RMA and
    silently returned Wilder values for mamode='sma'.
    """
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    ind_res = atr_ind(
        high, low, close, length=10, mamode='sma', use_talib=True
    )
    numba_res = atr_numba(high, low, close, length=10, mamode='sma')
    assert_allclose(ind_res, numba_res, rtol=1e-12, equal_nan=True)


@pytest.mark.volatility
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_atr_ind_forces_numba_for_drift_two(
    df_ohlc: pl.DataFrame,
) -> None:
    """atr_ind(drift=2) must use Numba even with use_talib=True."""
    high = df_ohlc['high'].to_numpy()
    low = df_ohlc['low'].to_numpy()
    close = df_ohlc['close'].to_numpy()
    ind_res = atr_ind(high, low, close, length=10, drift=2, use_talib=True)
    numba_res = atr_numba(high, low, close, length=10, drift=2)
    assert_allclose(ind_res, numba_res, rtol=1e-12, equal_nan=True)


# -----------------------------------------------------------------------------
# Input flexibility: Polars Series, integer inputs
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_atr_ind_accepts_polars_series(df_ohlc: pl.DataFrame) -> None:
    """atr_ind accepts pl.Series and matches the numpy result."""
    ind_res = atr_ind(
        df_ohlc['high'], df_ohlc['low'], df_ohlc['close'], length=10
    )
    numba_res = atr_numba(
        df_ohlc['high'].to_numpy(),
        df_ohlc['low'].to_numpy(),
        df_ohlc['close'].to_numpy(),
        length=10,
    )
    assert_allclose(ind_res, numba_res, rtol=1e-12, equal_nan=True)


@pytest.mark.volatility
def test_atr_accepts_integer_prices(df_ohlc: pl.DataFrame) -> None:
    """Integer price arrays are cast to float64 internally."""
    high = (df_ohlc['high'] * 100).round().cast(pl.Int64)
    low = (df_ohlc['low'] * 100).round().cast(pl.Int64)
    close = (df_ohlc['close'] * 100).round().cast(pl.Int64)
    result = atr_numba(
        high.to_numpy(), low.to_numpy(), close.to_numpy(), length=10
    )
    expected = atr_numba(
        high.cast(pl.Float64).to_numpy(),
        low.cast(pl.Float64).to_numpy(),
        close.cast(pl.Float64).to_numpy(),
        length=10,
    )
    assert result.dtype == np.float64
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


# -----------------------------------------------------------------------------
# Polars DataFrame integration
# -----------------------------------------------------------------------------
@pytest.mark.volatility
def test_atr_polars_adds_column(df_ohlc: pl.DataFrame) -> None:
    """atr_polars adds an ATR_<length> column of full length."""
    result = atr_polars(df_ohlc, length=10)
    assert 'ATR_10' in result.columns
    assert result.height == df_ohlc.height
    atr_col = result['ATR_10'].to_numpy()
    assert np.isnan(atr_col[:10]).all()  # warm-up = drift + length - 1
    assert np.isfinite(atr_col[10:]).all()


@pytest.mark.volatility
def test_atr_polars_does_not_mutate_input(df_ohlc: pl.DataFrame) -> None:
    """The input DataFrame is not modified in place."""
    before = df_ohlc.columns
    atr_polars(df_ohlc, length=10)
    assert df_ohlc.columns == before
    assert 'ATR_10' not in df_ohlc.columns


@pytest.mark.volatility
def test_atr_polars_custom_output_col(df_ohlc: pl.DataFrame) -> None:
    """output_col overrides the default ATR_<length> column name."""
    result = atr_polars(df_ohlc, length=10, output_col='my_atr')
    assert 'my_atr' in result.columns
    assert 'ATR_10' not in result.columns


@pytest.mark.volatility
def test_atr_polars_matches_numpy(df_ohlc: pl.DataFrame) -> None:
    """atr_polars values match atr_ind on the same data."""
    df_result = atr_polars(df_ohlc, length=10, mamode='sma')
    numpy_res = atr_ind(
        df_ohlc['high'].to_numpy(),
        df_ohlc['low'].to_numpy(),
        df_ohlc['close'].to_numpy(),
        length=10,
        mamode='sma',
    )
    assert_allclose(
        df_result['ATR_10'].to_numpy(), numpy_res,
        rtol=1e-12, equal_nan=True,
    )


@pytest.mark.volatility
def test_atr_polars_custom_columns() -> None:
    """Custom high/low/close column names are honoured."""
    n = 60
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pl.DataFrame({
        'h': close + np.abs(rng.normal(0, 0.8, n)),
        'l': close - np.abs(rng.normal(0, 0.8, n)),
        'c': close,
    })
    result = atr_polars(
        df, high_col='h', low_col='l', close_col='c', length=10
    )
    assert 'ATR_10' in result.columns
    assert np.isfinite(result['ATR_10'].to_numpy()[10:]).all()