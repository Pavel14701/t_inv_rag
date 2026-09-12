# -*- coding: utf-8 -*-
"""Unit tests for the ADX module (trend strength indicator).

Tests cover:
- output contract: shapes, dtypes, column semantics (adx, adxr, dmp, dmn)
- value ranges: ADX/ADXR in [0, scalar], DMP/DMN >= 0
- warm-up NaN region (first ``length + signal_length - 2`` bars)
- directional semantics on constructed trends (up-trend: DMP > DMN)
- ADXR identity: 0.5 * (adx + adx shifted by adxr_length)
- tvmode vs standard mode (both valid, different smoothing)
- talib vs pure-numpy parity (approximate, when TA-Lib is available)
- trim mode shortening
- input validation (lengths, inf inputs, too-short arrays)
- adx_ind (Polars Series input) and adx_polars integration
"""

import numpy as np
import polars as pl
import pytest

from ...external import talib_available
from ...trend.adx import adx_ind, adx_numpy, adx_polars


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
LENGTH = 14
SIGNAL_LENGTH = 14
WARMUP = LENGTH + SIGNAL_LENGTH - 1  # first index with a valid ADX value


def _make_ohlc(
    n: int = 200,
    drift: float = 0.2,
    volatility: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic high/low/close built around a random walk."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(drift, volatility, n))
    spread = np.abs(rng.normal(0.5, 0.2, n))
    high = close + spread
    low = close - spread
    return high, low, close


@pytest.fixture
def ohlc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _make_ohlc()


# -----------------------------------------------------------------------------
# Output contract
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_adx_output_shapes(ohlc) -> None:
    high, low, close = ohlc
    adx, adxr, dmp, dmn = adx_numpy(high, low, close, length=LENGTH)
    for arr in (adx, adxr, dmp, dmn):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert len(arr) == len(close)


@pytest.mark.trend
def test_adx_warmup_nan_region(ohlc) -> None:
    high, low, close = ohlc
    adx, adxr, dmp, dmn = adx_numpy(
        high, low, close, length=LENGTH, use_talib=False,
    )
    first_adx = LENGTH + SIGNAL_LENGTH - 1
    # DMP/DMN valid from index length, ADX from length+signal_length-1
    assert np.all(np.isnan(dmp[:LENGTH]))
    assert np.all(np.isnan(dmn[:LENGTH]))
    assert np.all(np.isnan(adx[:first_adx]))
    assert np.all(np.isnan(adxr[:first_adx + 1]))  # adxr lags one more bar
    assert np.isfinite(adx[first_adx])
    assert np.isfinite(dmp[-1])


@pytest.mark.trend
def test_adx_tvmode_not_all_nan(ohlc) -> None:
    """Regression: tvmode recursion must not be seeded with warm-up NaN.

    Previously ``dmp[length-1]`` used ``k[length-1]`` where ATR was still
    NaN, poisoning the recursive smoothing - tvmode output was NaN
    everywhere.
    """
    high, low, close = ohlc
    adx_tv, _, dmp_tv, dmn_tv = adx_numpy(
        high, low, close, length=LENGTH, tvmode=True, use_talib=False,
    )
    assert np.isfinite(dmp_tv[LENGTH:]).all()
    assert np.isfinite(dmn_tv[LENGTH:]).all()
    # tvmode shifts DX back by `length` bars (TradingView semantics),
    # so the very tail of ADX is undefined by design
    tail = len(adx_tv) - LENGTH
    assert np.isfinite(adx_tv[LENGTH + SIGNAL_LENGTH - 1:tail]).all()
    # tvmode DMP/DMN should be in the same ballpark as the standard
    # ones once their different initialisations have washed out
    _, _, dmp_std, _ = adx_numpy(
        high, low, close, length=LENGTH, tvmode=False, use_talib=False,
    )
    np.testing.assert_allclose(
        np.nanmedian(dmp_tv[-100:]), np.nanmedian(dmp_std[-100:]),
        rtol=0.1,
    )


# -----------------------------------------------------------------------------
# Value ranges and semantics
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_adx_value_ranges(ohlc) -> None:
    high, low, close = ohlc
    adx, adxr, dmp, dmn = adx_numpy(high, low, close, length=LENGTH)
    assert np.nanmin(adx) >= 0.0
    assert np.nanmax(adx) <= 100.0
    assert np.nanmin(adxr) >= 0.0
    assert np.nanmax(adxr) <= 100.0
    assert np.nanmin(dmp) >= 0.0
    assert np.nanmin(dmn) >= 0.0


@pytest.mark.trend
def test_adx_uptrend_dmp_dominates() -> None:
    """In a steady up-trend DMP must exceed DMN and ADX must be strong."""
    n = 150
    close = 100.0 + np.arange(n, dtype=np.float64) * 0.5
    high = close + 0.5
    low = close - 0.5
    adx, adxr, dmp, dmn = adx_numpy(
        high, low, close, length=LENGTH, use_talib=False,
    )
    tail = slice(WARMUP, n)
    assert np.all(dmp[tail] > dmn[tail])
    assert np.nanmean(adx[tail]) > 25.0


@pytest.mark.trend
def test_adx_downtrend_dmn_dominates() -> None:
    n = 150
    close = 200.0 - np.arange(n, dtype=np.float64) * 0.5
    high = close + 0.5
    low = close - 0.5
    adx, adxr, dmp, dmn = adx_numpy(
        high, low, close, length=LENGTH, use_talib=False,
    )
    tail = slice(WARMUP, n)
    assert np.all(dmn[tail] > dmp[tail])
    assert np.nanmean(adx[tail]) > 25.0


# -----------------------------------------------------------------------------
# ADXR identity
# -----------------------------------------------------------------------------
@pytest.mark.trend
@pytest.mark.parametrize('adxr_length', [2, 3, 5])
def test_adxr_identity(ohlc, adxr_length: int) -> None:
    high, low, close = ohlc
    adx, adxr, _, _ = adx_numpy(
        high, low, close, length=LENGTH, adxr_length=adxr_length,
        use_talib=False,
    )
    expected = 0.5 * (adx + np.roll(adx, adxr_length))
    expected[:adxr_length] = np.nan
    mask = np.isfinite(adxr)
    np.testing.assert_allclose(adxr[mask], expected[mask])


# -----------------------------------------------------------------------------
# Modes: tvmode and talib parity
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_adx_tvmode_valid_and_different(ohlc) -> None:
    high, low, close = ohlc
    adx_std, _, dmp_std, dmn_std = adx_numpy(
        high, low, close, length=LENGTH, tvmode=False, use_talib=False,
    )
    adx_tv, _, dmp_tv, dmn_tv = adx_numpy(
        high, low, close, length=LENGTH, tvmode=True, use_talib=False,
    )
    assert np.isfinite(adx_tv[WARMUP:-LENGTH]).all()
    assert np.isfinite(dmp_tv[LENGTH:]).all()
    assert np.nanmin(adx_tv) >= 0.0
    assert np.nanmax(adx_tv) <= 100.0
    # different smoothing must produce different series (not a copy)
    assert not np.allclose(adx_std[WARMUP:-LENGTH], adx_tv[WARMUP:-LENGTH])
    assert not np.allclose(dmp_std[LENGTH - 1:], dmp_tv[LENGTH - 1:])


@pytest.mark.trend
@pytest.mark.skipif(not talib_available, reason='TA-Lib not installed')
def test_adx_talib_parity(ohlc) -> None:
    """use_talib=True (TA-Lib ADX) ~= pure-numpy rma path (both Wilder).

    The two backends initialise their Wilder smoothing differently, so
    they are only compared after the transients have decayed (the rma
    contraction factor (1 - 1/length)^k makes 6*length bars ample).
    """
    high, low, close = ohlc
    adx_talib, _, _, _ = adx_numpy(
        high, low, close, length=LENGTH, use_talib=True,
    )
    adx_np, _, _, _ = adx_numpy(
        high, low, close, length=LENGTH, use_talib=False,
    )
    settled = LENGTH * 6
    np.testing.assert_allclose(
        adx_talib[settled:], adx_np[settled:], atol=1e-6,
    )
    np.testing.assert_allclose(
        np.nanmedian(adx_talib[-100:]), np.nanmedian(adx_np[-100:]),
        atol=1e-6,
    )


# -----------------------------------------------------------------------------
# Trim, scalar
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_adx_trim_shortens(ohlc) -> None:
    high, low, close = ohlc
    n = len(close)
    adx, adxr, dmp, dmn = adx_numpy(
        high, low, close, length=LENGTH, trim=True, use_talib=False,
    )
    expected = n - (LENGTH + SIGNAL_LENGTH - 1)
    assert len(adx) == expected
    assert np.all(np.isfinite(adx))
    assert np.all(np.isfinite(dmp))
    assert np.all(np.isfinite(dmn))
    # trimmed output must equal the tail of the untrimmed one
    full, _, _, _ = adx_numpy(
        high, low, close, length=LENGTH, trim=False, use_talib=False,
    )
    np.testing.assert_array_equal(adx, full[-expected:])


@pytest.mark.trend
def test_adx_custom_scalar(ohlc) -> None:
    high, low, close = ohlc
    adx, _, _, _ = adx_numpy(
        high, low, close, length=LENGTH, scalar=1.0, use_talib=False,
    )
    assert np.nanmax(adx) <= 1.0
    assert np.nanmin(adx) >= 0.0


# -----------------------------------------------------------------------------
# Input validation and edge cases
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_adx_invalid_length_raises(ohlc) -> None:
    high, low, close = ohlc
    with pytest.raises(ValueError, match='length'):
        adx_numpy(high, low, close, length=0)
    with pytest.raises(ValueError, match='adxr_length'):
        adx_numpy(high, low, close, adxr_length=0)


@pytest.mark.trend
def test_adx_inf_input_raises(ohlc) -> None:
    high, low, close = ohlc
    low_bad = low.copy()
    low_bad[10] = np.inf
    with pytest.raises(ValueError, match='low'):
        adx_numpy(high, low_bad, close)


@pytest.mark.trend
def test_adx_too_short_raises() -> None:
    high = np.array([1.0, 2.0, 3.0])
    low = np.array([0.9, 1.9, 2.9])
    close = np.array([1.0, 2.0, 2.8])
    with pytest.raises(ValueError):
        adx_numpy(high, low, close, length=LENGTH)


@pytest.mark.trend
def test_adx_uses_default_signal_length(ohlc) -> None:
    high, low, close = ohlc
    a1, _, _, _ = adx_numpy(
        high, low, close, length=LENGTH, signal_length=None, use_talib=False,
    )
    a2, _, _, _ = adx_numpy(
        high, low, close, length=LENGTH, signal_length=LENGTH,
        use_talib=False,
    )
    np.testing.assert_array_equal(a1, a2)


# -----------------------------------------------------------------------------
# Wrappers
# -----------------------------------------------------------------------------
@pytest.mark.trend
def test_adx_ind_accepts_polars_series(ohlc) -> None:
    high, low, close = ohlc
    res_np = adx_ind(high, low, close, length=LENGTH, use_talib=False)
    res_pl = adx_ind(
        pl.Series(high), pl.Series(low), pl.Series(close),
        length=LENGTH, use_talib=False,
    )
    for a, b in zip(res_np, res_pl):
        np.testing.assert_array_equal(a, b)


@pytest.mark.trend
def test_adx_polars_columns(ohlc) -> None:
    high, low, close = ohlc
    df = pl.DataFrame({
        'date': np.arange(len(close)),
        'high': high, 'low': low, 'close': close,
    })
    out = adx_polars(df, length=LENGTH)
    assert out.height == df.height
    for col in ('ADX_14', 'ADXR_14_2', 'DMP_14', 'DMN_14'):
        assert col in out.columns
    assert out['ADX_14'].dtype == pl.Float64
    assert np.isfinite(out['ADX_14'].to_numpy()[WARMUP])
