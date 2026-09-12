"""Unit tests for Heikin-Ashi calculation (Numba implementation)."""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.candle.heikin_ashi import ha, ha_numpy, ha_polars


@pytest.mark.unit
@pytest.mark.candle
def test_heikin_ashi_basic():
    """Test Heikin-Ashi calculation on a simple input."""
    open_ = np.array([100, 102, 104, 106], dtype=np.float64)
    high = np.array([101, 103, 105, 107], dtype=np.float64)
    low = np.array([99, 101, 103, 105], dtype=np.float64)
    close = np.array([102, 104, 106, 108], dtype=np.float64)

    # Expected HA values (calculated manually)
    expected_ha_close = np.array([100.5, 102.5, 104.5, 106.5])
    expected_ha_open = np.array([101.0, 100.75, 101.625, 103.0625])
    expected_ha_high = np.array([101.0, 103.0, 105.0, 107.0])
    expected_ha_low = np.array([99.0, 100.75, 101.625, 103.0625])

    ha_o, ha_h, ha_l, ha_c = ha_numpy(open_, high, low, close)

    assert_allclose(ha_o, expected_ha_open, rtol=1e-6)
    assert_allclose(ha_h, expected_ha_high, rtol=1e-6)
    assert_allclose(ha_l, expected_ha_low, rtol=1e-6)
    assert_allclose(ha_c, expected_ha_close, rtol=1e-6)


@pytest.mark.unit
@pytest.mark.candle
def test_heikin_ashi_offset_fillna():
    """Test HA offset and fillna parameters."""
    open_ = np.array([100, 102, 104], dtype=np.float64)
    high = np.array([101, 103, 105], dtype=np.float64)
    low = np.array([99, 101, 103], dtype=np.float64)
    close = np.array([102, 104, 106], dtype=np.float64)
    # Without offset (reference)
    ha_o0, ha_h0, ha_l0, ha_c0 = ha_numpy(open_, high, low, close)
    # offset=1, fillna=0.0
    ha_o, ha_h, ha_l, ha_c = ha_numpy(
        open_, high, low, close, offset=1, fillna=0.0
    )
    # First element should be fillna value (0.0)
    assert ha_o[0] == 0.0
    assert ha_h[0] == 0.0
    assert ha_l[0] == 0.0
    assert ha_c[0] == 0.0
    # Check shifted values: offset=1 means index i gets value from index i-1
    assert_allclose(ha_o[1:], ha_o0[:-1])
    assert_allclose(ha_h[1:], ha_h0[:-1])
    assert_allclose(ha_l[1:], ha_l0[:-1])
    assert_allclose(ha_c[1:], ha_c0[:-1])


@pytest.mark.unit
@pytest.mark.candle
def test_heikin_ashi_with_pl_series():
    """Test HA with Polars Series input."""
    open_s = pl.Series([100, 102, 104, 106])
    high_s = pl.Series([101, 103, 105, 107])
    low_s = pl.Series([99, 101, 103, 105])
    close_s = pl.Series([102, 104, 106, 108])
    ha_o, ha_h, ha_l, ha_c = ha(open_s, high_s, low_s, close_s)
    # Check types and shapes
    assert isinstance(ha_o, np.ndarray)
    assert ha_o.shape == (4,)
    assert ha_o.dtype == np.float64
    # Basic correctness: last value should be > previous etc.
    # We'll just check that it's not all zeros
    assert not np.allclose(ha_o, 0.0)


@pytest.mark.unit
@pytest.mark.candle
def test_heikin_ashi_polars_basic():
    """Test HA Polars wrapper returns correct DataFrame."""
    df = pl.DataFrame(
        {
            "date": [1, 2, 3, 4],
            "open": [100, 102, 104, 106],
            "high": [101, 103, 105, 107],
            "low": [99, 101, 103, 105],
            "close": [102, 104, 106, 108],
        }
    )
    result = ha_polars(df, suffix="_HA")
    expected_cols = [
        "date",
        "HA_open_HA",
        "HA_high_HA",
        "HA_low_HA",
        "HA_close_HA",
    ]
    assert list(result.columns) == expected_cols
    assert len(result) == len(df)
    # Check that HA columns are float64
    for col in expected_cols[1:]:
        assert result[col].dtype == pl.Float64
    # Check that the values are the same as from ha_numpy (skip date col)
    open_arr = df["open"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()
    ha_o_np, ha_h_np, ha_l_np, ha_c_np = ha_numpy(
        open_arr, high_arr, low_arr, close_arr
    )
    assert_allclose(result["HA_open_HA"].to_numpy(), ha_o_np)
    assert_allclose(result["HA_high_HA"].to_numpy(), ha_h_np)
    assert_allclose(result["HA_low_HA"].to_numpy(), ha_l_np)
    assert_allclose(result["HA_close_HA"].to_numpy(), ha_c_np)


@pytest.mark.unit
@pytest.mark.candle
def test_heikin_ashi_polars_with_offset_fillna():
    """Test HA Polars with offset and fillna."""
    df = pl.DataFrame(
        {
            "date": [1, 2, 3, 4],
            "open": [100, 102, 104, 106],
            "high": [101, 103, 105, 107],
            "low": [99, 101, 103, 105],
            "close": [102, 104, 106, 108],
        }
    )
    result = ha_polars(df, offset=1, fillna=0.0, suffix="_HA")
    assert result["HA_open_HA"][0] == 0.0
    assert result["HA_high_HA"][0] == 0.0
    assert result["HA_low_HA"][0] == 0.0
    assert result["HA_close_HA"][0] == 0.0
    # Compare shifted values
    result_no_offset = ha_polars(df, suffix="_HA")
    for col in ["HA_open_HA", "HA_high_HA", "HA_low_HA", "HA_close_HA"]:
        assert_allclose(
            result[col].to_numpy()[1:], result_no_offset[col].to_numpy()[:-1]
        )
