# -*- coding: utf-8 -*-
"""Unit tests for Holt-Winter Moving Average (HWMA) module.

Tests cover:
- _hwma_numba_core against reference implementation
- parameter validation (na/nb/nc)
- offset and fillna
- hwma_ind with Polars Series
- hwma_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.hwma import (
    _hwma_numba_core,
    hwma_numba,
    hwma_ind,
    hwma_polars,
)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------

def _hwma_reference(
    close: npt.NDArray[np.float64],
    na: float,
    nb: float,
    nc: float,
) -> npt.NDArray[np.float64]:
    """Pure Python reference implementation of HWMA."""
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    last_a = 0.0
    last_v = 0.0
    last_f = close[0]
    for i in range(n):
        f = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * close[i]
        v = (1.0 - nb) * (last_v + last_a) + nb * (f - last_f)
        a = (1.0 - nc) * last_a + nc * (v - last_v)
        out[i] = f + v + 0.5 * a
        last_a, last_f, last_v = a, f, v
    return out


# -----------------------------------------------------------------------------
# Tests for _hwma_numba_core
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_hwma_core_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test _hwma_numba_core against pure Python reference."""
    close = prices_random_walk
    result = _hwma_numba_core(close, 0.2, 0.1, 0.1)
    expected = _hwma_reference(close, 0.2, 0.1, 0.1)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hwma_core_empty() -> None:
    """Empty input returns empty array (no IndexError)."""
    result = _hwma_numba_core(np.array([]), 0.2, 0.1, 0.1)
    assert result.size == 0


# -----------------------------------------------------------------------------
# Tests for hwma_numba
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_hwma_numba_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test hwma_numba against pure Python reference."""
    close = prices_random_walk
    result = hwma_numba(close, na=0.2, nb=0.1, nc=0.1)
    expected = _hwma_reference(close, 0.2, 0.1, 0.1)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hwma_numba_invalid_parameters() -> None:
    """Invalid na/nb/nc raise ValueError instead of silent defaults."""
    close = np.arange(1.0, 11.0)
    with pytest.raises(ValueError, match='na'):
        hwma_numba(close, na=0.0)
    with pytest.raises(ValueError, match='na'):
        hwma_numba(close, na=1.0)
    with pytest.raises(ValueError, match='nb'):
        hwma_numba(close, nb=-0.1)
    with pytest.raises(ValueError, match='nc'):
        hwma_numba(close, nc=1.5)


@pytest.mark.overlap
def test_hwma_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna behaviour."""
    close = prices_random_walk
    base = hwma_numba(close, offset=0, fillna=None)
    result = hwma_numba(close, offset=2, fillna=0.0)
    # HWMA has no warmup NaN: base is finite everywhere
    assert np.isfinite(base).all()
    # positive offset shifts forward, fillna fills shifted-in positions
    assert_allclose(result[2:], base[:-2], rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hwma_numba_fillna_all_nan_input() -> None:
    """All NaN input with fillna replaces everything."""
    data = np.full(10, np.nan)
    result = hwma_numba(data, fillna=0.0, nan_policy='ignore')
    assert (result == 0.0).all()


# -----------------------------------------------------------------------------
# Tests for hwma_ind (universal wrapper)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_hwma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test hwma_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    result = hwma_ind(s, na=0.2, nb=0.1, nc=0.1)
    expected = _hwma_reference(prices_random_walk, 0.2, 0.1, 0.1)
    assert_allclose(result, expected, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for hwma_polars (DataFrame integration)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_hwma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test hwma_polars adds a column correctly."""
    result_df = hwma_polars(
        df_random_walk, na=0.2, nb=0.1, nc=0.1, output_col='HWMA'
    )
    assert 'HWMA' in result_df.columns
    assert result_df['HWMA'].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk['close'].to_numpy()
    expected = _hwma_reference(close_arr, 0.2, 0.1, 0.1)
    assert_allclose(
        result_df['HWMA'].to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_hwma_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame(
        {'close': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
    )
    result_df = hwma_polars(df)
    assert 'HWMA_0.2_0.1_0.1' in result_df.columns


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_hwma_numba_nan_policy_raise() -> None:
    """Input with NaN and default nan_policy='raise' raises ValueError."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match='NaN'):
        hwma_numba(data)


@pytest.mark.overlap
def test_hwma_numba_nan_policy_ffill() -> None:
    """Input with NaN and nan_policy='ffill' is filled and computed."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    result = hwma_numba(data, nan_policy='ffill')
    assert np.isfinite(result).all()


@pytest.mark.overlap
def test_hwma_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    with pytest.raises(ValueError, match='NaN'):
        hwma_numba(prices_with_inf)
    result = hwma_numba(prices_with_inf, nan_policy='ffill')
    assert np.isfinite(result).all()


@pytest.mark.overlap
def test_hwma_numba_empty(prices_empty):
    """Empty input returns empty array."""
    result = hwma_numba(prices_empty)
    assert result.size == 0


@pytest.mark.overlap
def test_hwma_numba_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    result = hwma_numba(prices_extreme, nan_policy='ignore')
    assert result is not None