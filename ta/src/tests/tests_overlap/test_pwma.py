# -*- coding: utf-8 -*-
"""Unit tests for PWMA (Pascal's Weighted Moving Average) module.

Tests cover:
- _pascal_weights: binomial coefficients, normalisation, symmetry
- _pwma_numba_core / pwma_numba against a pure-Python reference
- offset and fillna
- Input validation (length < 1)
- Universal wrapper (pwma_ind) with Polars Series and list input
- Polars integration (pwma_polars)
- IEEE 754 compliance (NaN, Inf, empty, short, all-NaN, extreme)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose
from math import comb

from ...overlap.pwma import (
    _pascal_weights,
    _pwma_numba_core,
    pwma_numba,
    pwma_ind,
    pwma_polars,
)
from ..._array_ops import _apply_offset_fillna


# -----------------------------------------------------------------------------
# Reference implementations
# -----------------------------------------------------------------------------

def _pascal_weights_reference(length: int) -> np.ndarray:
    """Normalized binomial coefficients of Pascal row (length-1)."""
    w = np.array(
        [comb(length - 1, k) for k in range(length)],
        dtype=np.float64
    )
    return w / w.sum()


def _pwma_reference(
    close: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure-Python PWMA reference."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    w = _pascal_weights_reference(length)
    for i in range(length - 1, n):
        out[i] = float(np.dot(close[i - length + 1:i + 1], w))
    return out


# -----------------------------------------------------------------------------
# Weights tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_pascal_weights_match_binomials() -> None:
    """Weights equal normalized binomial coefficients for various lengths."""
    for length in (1, 2, 3, 5, 10, 20):
        w = _pascal_weights(length, True)
        assert_allclose(w, _pascal_weights_reference(length), rtol=1e-12)
        assert w.sum() == pytest.approx(1.0)


@pytest.mark.overlap
def test_pascal_weights_symmetric() -> None:
    """Pascal rows are symmetric: asc and desc weights are identical."""
    for length in (2, 3, 5, 8, 13):
        assert np.array_equal(_pascal_weights(length, True),
                              _pascal_weights(length, False))


@pytest.mark.overlap
def test_pascal_weights_cached_readonly() -> None:
    """Cached weight arrays must not be writable (lru_cache safety)."""
    w1 = _pascal_weights(7, True)
    assert not w1.flags.writeable
    w2 = _pascal_weights(7, True)
    assert w2 is w1  # same cached object
    assert_allclose(w2, _pascal_weights_reference(7), rtol=1e-12)


# -----------------------------------------------------------------------------
# Core / pwma_numba tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_pwma_numba_core_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Compare the numba core with the pure-Python reference."""
    for length in (2, 5, 10):
        result = _pwma_numba_core(
            prices_random_walk,
            _pascal_weights(length, True)
        )
        expected = _pwma_reference(prices_random_walk, length)
        assert result.shape == prices_random_walk.shape
        assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
        assert np.isnan(result[:length - 1]).all()


@pytest.mark.overlap
def test_pwma_hand_computed() -> None:
    """Hand-computed PWMA with length=3, weights [1, 2, 1] / 4."""
    close = np.arange(1.0, 7.0)  # 1..6
    result = pwma_numba(close, length=3)
    # out[i] = (close[i-2] + 2*close[i-1] + close[i]) / 4
    assert_allclose(result[2:], [(1 + 4 + 3) / 4, (2 + 6 + 4) / 4,
                                 (3 + 8 + 5) / 4, (4 + 10 + 6) / 4],
                    rtol=1e-12)
    assert np.isnan(result[:2]).all()


@pytest.mark.overlap
def test_pwma_length_one_is_identity() -> None:
    """length=1 -> weight [1.0] -> PWMA equals the close series."""
    close = np.arange(1.0, 8.0)
    result = pwma_numba(close, length=1)
    assert_allclose(result, close, rtol=1e-12)


@pytest.mark.overlap
def test_pwma_constant_series() -> None:
    """Constant series -> PWMA equals the constant (weights sum to 1)."""
    close = np.full(20, 42.0)
    result = pwma_numba(close, length=5)
    assert_allclose(result[4:], 42.0, rtol=1e-12)


@pytest.mark.overlap
def test_pwma_asc_desc_identical(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """asc flag is a no-op: symmetric weights give identical output."""  # noqa: D403
    ra = pwma_numba(prices_random_walk, length=8, asc=True)
    rb = pwma_numba(prices_random_walk, length=8, asc=False)
    mask = ~np.isnan(ra)
    assert_allclose(ra[mask], rb[mask], rtol=1e-12)


@pytest.mark.overlap
def test_pwma_offset_fillna() -> None:
    """Test pwma_numba with offset and fillna."""
    close = np.arange(1.0, 11.0)
    offset = 2
    fillna = 0.0
    base = pwma_numba(close, length=3, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = pwma_numba(close, length=3, offset=offset, fillna=fillna)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
    assert (result[:offset] == fillna).all()


@pytest.mark.overlap
def test_pwma_invalid_length() -> None:
    """length < 1 must raise ValueError, never corrupt the output."""
    close = np.array([10.0, 11.0, 12.0])
    with pytest.raises(ValueError, match='length must be >= 1'):
        pwma_numba(close, length=0)
    with pytest.raises(ValueError, match='length must be >= 1'):
        pwma_numba(close, length=-2)


@pytest.mark.overlap
def test_pwma_input_types() -> None:
    """float32 input, Python list and read-only arrays are handled."""
    r32 = pwma_numba(np.arange(1.0, 6.0, dtype=np.float32), length=2)
    assert r32.dtype == np.float64
    r_list = pwma_ind([1.0, 2.0, 3.0, 4.0], length=3)
    assert_allclose(r_list[2:], [2.0, 3.0], rtol=1e-12)
    c = np.arange(1.0, 6.0)
    c.setflags(write=False)
    assert np.isfinite(pwma_numba(c, length=3)[2:]).all()


# -----------------------------------------------------------------------------
# Universal wrapper tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_pwma_ind_matches_numba(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """pwma_ind returns the same result as pwma_numba."""
    result = pwma_ind(prices_random_walk, length=10)
    expected = pwma_numba(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.overlap
def test_pwma_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """pwma_ind accepts a Polars Series and matches pwma_numba."""
    s = pl.Series(prices_random_walk)
    result = pwma_ind(s, length=10)
    expected = pwma_numba(prices_random_walk, length=10)
    assert_allclose(result, expected, rtol=1e-12)

# -----------------------------------------------------------------------------
# Polars integration tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_pwma_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """pwma_polars returns a DataFrame with a correct PWMA column."""
    length = 10
    result = pwma_polars(df_random_walk, length=length)
    assert isinstance(result, pl.DataFrame)
    assert f'PWMA_{length}' in result.columns
    close_arr = df_random_walk['close'].to_numpy()
    expected = _pwma_reference(close_arr, length)
    mask = ~np.isnan(expected)
    assert_allclose(
        result[f'PWMA_{length}'].to_numpy()[mask], expected[mask], rtol=1e-12,
    )


@pytest.mark.overlap
def test_pwma_polars_custom_output_col(df_random_walk) -> None:
    """Custom output column name is respected."""
    result = pwma_polars(df_random_walk, length=5, output_col='PWMA')
    assert 'PWMA' in result.columns
    assert result['PWMA'].dtype == pl.Float64


@pytest.mark.overlap
def test_pwma_polars_custom_close_col(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """pwma_polars with a non-default close column name."""
    df = pl.DataFrame({'price': prices_random_walk})
    result = pwma_polars(df, close_col='price', length=10, output_col='PWMA')
    expected = _pwma_reference(prices_random_walk, 10)
    mask = ~np.isnan(expected)
    assert_allclose(result['PWMA'].to_numpy()[mask], expected[mask])


@pytest.mark.overlap
def test_pwma_polars_with_offset_fillna(df_random_walk) -> None:
    """pwma_polars applies offset and fillna."""
    offset = 2
    fillna = 0.0
    close_arr = df_random_walk['close'].to_numpy()
    base = pwma_numba(close_arr, length=5, offset=0, fillna=None)
    expected = _apply_offset_fillna(base, offset, fillna)
    result = pwma_polars(
        df_random_walk, length=5, offset=offset, fillna=fillna,
        output_col='PWMA',
    )
    assert_allclose(result['PWMA'].to_numpy(), expected, rtol=1e-12)


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_pwma_with_nan(prices_with_nan) -> None:
    """NaN poisons only the windows that contain it (no recursion)."""
    result = pwma_numba(prices_with_nan, length=3)
    # NaN at index 5 poisons windows starting at i=5,6 (indices 3..5, 4..6).
    assert np.isfinite(result[2:5]).all()
    assert np.isnan(result[5:7]).all()
    assert np.isfinite(result[8:]).all()


@pytest.mark.overlap
def test_pwma_with_inf(prices_with_inf) -> None:
    """Inf propagates through the weighted sum naturally (no crash)."""
    result = pwma_numba(prices_with_inf, length=3)
    assert result is not None
    # Windows containing the Inf at index 5 yield Inf (i=5,6).
    assert np.isinf(result[5:7]).all()
    assert np.isfinite(result[2:5]).all()
    assert np.isfinite(result[8:]).all()


@pytest.mark.overlap
def test_pwma_empty(prices_empty) -> None:
    """Empty input returns an empty array."""
    result = pwma_numba(prices_empty, length=3)
    assert result.size == 0


@pytest.mark.overlap
def test_pwma_single(prices_single) -> None:
    """Single-element input: too short for length=2, so all NaN."""
    result = pwma_numba(prices_single, length=2)
    assert np.isnan(result).all()
    result1 = pwma_numba(prices_single, length=1)
    assert result1[0] == prices_single[0]


@pytest.mark.overlap
def test_pwma_all_nan(prices_all_nan) -> None:
    """All NaNs -> all NaNs (or fillna if provided)."""
    result = pwma_numba(prices_all_nan, length=3)
    assert np.isnan(result).all()
    result_fill = pwma_numba(prices_all_nan, length=3, fillna=0.0)
    assert (result_fill == 0.0).all()


@pytest.mark.overlap
def test_pwma_extreme_values(prices_extreme) -> None:
    """Extreme values must not crash."""
    result = pwma_numba(prices_extreme, length=3)
    assert result is not None


@pytest.mark.overlap
def test_pwma_polars_with_nan(df_random_walk: pl.DataFrame) -> None:
    """Polars integration propagates NaN correctly."""
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series('close', close_arr))
    result = pwma_polars(
        df_with_nan, length=3, output_col='PWMA',
    )
    vals = result['PWMA'].to_numpy()
    nb = pwma_numba(close_arr, length=3)
    assert np.array_equal(
        np.nan_to_num(vals, nan=-999.0), np.nan_to_num(nb, nan=-999.0),
    )
    assert np.isfinite(vals[8:]).all()
