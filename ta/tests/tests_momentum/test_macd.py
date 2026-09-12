"""Unit tests for the MACD implementation.

These tests reuse shared fixtures defined in conftest.py:
    - price series: prices_uptrend, prices_downtrend, prices_sideways,
        prices_up_then_down, prices_down_then_up, prices_random_walk,
        prices_volatile, prices_with_reversals
    - DataFrames: df_uptrend, df_downtrend, df_sideways, df_random_walk,
        df_volatile
    - Common parameters: offset, fillna

All floating-point operations are tested for IEEE 754 compliance
(no crashes on NaN/Inf, empty arrays, extreme values).
"""
import pytest
import numpy as np
import polars as pl

from ...momentum.macd import macd_numpy, macd_ind, macd_polars


# MACD parameters used in tests
FAST = 12
SLOW = 26
SIGNAL = 9
# Index from which all values should be finite (slow + signal - 1)
START_IDX = SLOW + SIGNAL - 1


# -----------------------------------------------------------------------------
# Tests for macd_numpy (NumPy-based core)
# -----------------------------------------------------------------------------

def test_macd_numpy_basic(prices_random_walk):
    """Basic calculation: output shapes match input,
    values are finite after the initial NaN period.
    """
    macd, sig, hist = macd_numpy(prices_random_walk, use_talib=False)
    assert macd.shape == prices_random_walk.shape
    # After START_IDX, all values should be finite (not NaN)
    assert np.isfinite(macd[START_IDX:]).all()
    assert np.isfinite(sig[START_IDX:]).all()
    assert np.isfinite(hist[START_IDX:]).all()


def test_macd_numpy_with_talib(prices_uptrend):
    """Compare built-in calculation with TA-Lib (if available).
    Uses a simple uptrend to minimize numerical noise.
    Comparison is done only after the initial NaN period.
    """
    try:
        import talib  # noqa: F401
    except ImportError:
        pytest.skip('TA-Lib not available')
    macd1, sig1, hist1 = macd_numpy(prices_uptrend, use_talib=True)
    macd2, sig2, hist2 = macd_numpy(prices_uptrend, use_talib=False)
    # Compare only valid (non-NaN) portions
    np.testing.assert_allclose(macd1[START_IDX:], macd2[START_IDX:],
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(sig1[START_IDX:], sig2[START_IDX:],
                               rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(hist1[START_IDX:], hist2[START_IDX:],
                               rtol=1e-6, atol=1e-8)


def test_macd_numpy_asmode(prices_random_walk):
    """Verify AS mode: histogram equals macd - signal line after START_IDX."""
    macd, sig, hist = macd_numpy(
        prices_random_walk, asmode=True, use_talib=False
    )
    # Compare only valid portion
    np.testing.assert_array_almost_equal(
        hist[START_IDX:], macd[START_IDX:] - sig[START_IDX:], decimal=8
    )


def test_macd_numpy_offset_fillna(prices_random_walk, offset, fillna):
    """Check that offset and fillna affect only the first N elements."""
    macd, sig, hist = macd_numpy(prices_random_walk, offset=offset,
                                 fillna=fillna, use_talib=False)
    # First 'offset' elements must equal fillna
    np.testing.assert_array_equal(macd[:offset], fillna)
    np.testing.assert_array_equal(sig[:offset], fillna)
    np.testing.assert_array_equal(hist[:offset], fillna)
    # The rest should not all equal fillna
    assert not np.all(macd[offset:] == fillna)


def test_macd_numpy_empty():
    """Empty input must return empty arrays without errors."""
    empty = np.array([])
    macd, sig, hist = macd_numpy(empty)
    assert macd.size == 0
    assert sig.size == 0
    assert hist.size == 0


def test_macd_numpy_single():
    """Single-element input -> all NaNs (since series too short)."""
    single = np.array([10.0])
    macd, sig, hist = macd_numpy(single, use_talib=False)
    assert np.all(np.isnan(macd))
    assert np.all(np.isnan(sig))
    assert np.all(np.isnan(hist))


def test_macd_numpy_nan_inf():
    """IEEE 754: NaN and Inf propagate; Inf is converted to NaN."""
    data = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 6.0])
    macd, sig, hist = macd_numpy(data, use_talib=False)
    assert macd.shape == data.shape
    # NaN in input -> NaN in output
    assert np.isnan(macd[2])
    # Inf converted to NaN
    assert np.isnan(macd[4])


def test_macd_numpy_ieee754_extreme():
    """Extreme values (1e300, 1e-300) with length >= fast period."""
    extreme = np.array(
        [1e300, 1e-300, 1.0] * 10, dtype=np.float64
    )  # 30 elements
    macd, sig, hist = macd_numpy(extreme, use_talib=False)
    # Should not crash; may contain inf or nan, but that's allowed
    assert macd is not None


# -----------------------------------------------------------------------------
# Tests for macd_ind (universal wrapper)
# -----------------------------------------------------------------------------

def test_macd_ind_with_polars_series(prices_random_walk):
    """macd_ind should accept a Polars Series and return NumPy arrays."""
    series = pl.Series(prices_random_walk)
    macd, sig, hist = macd_ind(series, use_talib=False)
    assert isinstance(macd, np.ndarray)
    assert isinstance(sig, np.ndarray)
    assert isinstance(hist, np.ndarray)
    assert macd.shape == prices_random_walk.shape


# -----------------------------------------------------------------------------
# Tests for macd_polars (Polars DataFrame integration)
# -----------------------------------------------------------------------------

def test_macd_polars_adds_columns(df_random_walk):
    """macd_polars must add new columns to the original DataFrame."""
    result = macd_polars(df_random_walk, close_col='close')
    assert 'close' in result.columns
    assert 'MACD_12_26_9' in result.columns
    assert 'MACDs_12_26_9' in result.columns
    assert 'MACDh_12_26_9' in result.columns
    assert len(result) == len(df_random_walk)


def test_macd_polars_asmode_suffix(df_random_walk):
    """AS mode and custom suffix must produce correct column names."""
    result = macd_polars(df_random_walk, asmode=True, suffix='_test')
    assert 'MACDAS_test' in result.columns
    assert 'MACDASs_test' in result.columns
    assert 'MACDASh_test' in result.columns
    assert 'MACDAS_12_26_9' not in result.columns


def test_macd_polars_with_offset_fillna(df_random_walk, offset, fillna):
    """Integration test: offset and fillna are applied when adding columns."""
    result = macd_polars(df_random_walk, offset=offset, fillna=fillna)
    col = 'MACD_12_26_9'
    assert result[col].to_numpy()[:offset].tolist() == [fillna] * offset
    assert not np.all(result[col].to_numpy()[offset:] == fillna)


def test_macd_polars_works_with_different_patterns(
    df_uptrend, df_downtrend, df_sideways, df_volatile
):
    """Smoke test: works on various market patterns."""
    for df in (df_uptrend, df_downtrend, df_sideways, df_volatile):
        result = macd_polars(df, close_col='close')
        assert 'MACD_12_26_9' in result.columns
        assert len(result) == len(df)


# -----------------------------------------------------------------------------
# Additional edge-case tests using explicit data
# -----------------------------------------------------------------------------

def test_macd_numpy_all_identical():
    """All prices identical -> MACD, signal, histogram should be zero
    after the initial NaN period.
    """
    flat = np.full(100, 100.0, dtype=np.float64)
    macd, sig, hist = macd_numpy(flat, use_talib=False)
    # Skip the initial NaN period (up to START_IDX-1)
    np.testing.assert_array_almost_equal(macd[START_IDX:], 0.0, decimal=10)
    np.testing.assert_array_almost_equal(sig[START_IDX:], 0.0, decimal=10)
    np.testing.assert_array_almost_equal(hist[START_IDX:], 0.0, decimal=10)


def test_macd_numpy_negative_prices():
    """Negative prices should not break the calculation; values become
    finite after the initial NaN period.
    """
    neg = np.linspace(-10, 0, 50)
    macd, sig, hist = macd_numpy(neg, use_talib=False)
    # After START_IDX, all values should be finite
    assert np.isfinite(macd[START_IDX:]).all()
    assert np.isfinite(sig[START_IDX:]).all()
    assert np.isfinite(hist[START_IDX:]).all()
