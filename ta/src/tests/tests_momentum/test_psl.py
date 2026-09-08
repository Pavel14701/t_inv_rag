# -*- coding: utf-8 -*-
"""Unit tests for Psychological Line (PSL)."""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ...momentum.psl import _psl_numba, psl_ind, psl_numpy, psl_polars


def _psl_reference(
    close: np.ndarray, length: int = 12, drift: int = 1
) -> np.ndarray:
    """Pure numpy PSL reference."""
    n = len(close)
    out = np.full(n, np.nan)
    diff = np.full(n, np.nan)
    diff[drift:] = close[drift:] - close[:-drift]
    for i in range(length + drift - 1, n):
        window = diff[i - length + 1:i + 1]
        if np.isnan(window).any():
            continue
        out[i] = 100.0 * (window > 0).sum() / length
    return out


@pytest.mark.momentum
def test_psl_matches_reference(prices_random_walk) -> None:
    expected = _psl_reference(prices_random_walk, 12, 1)
    result = psl_numpy(prices_random_walk, length=12, drift=1)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_psl_warmup_nan() -> None:
    close = np.arange(1.0, 31.0)
    result = psl_numpy(close, length=12, drift=1)
    assert np.isnan(result[:12]).all()
    assert_allclose(result[12:], 100.0, rtol=1e-12)


@pytest.mark.momentum
def test_psl_bounds() -> None:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.standard_normal(100))
    result = psl_numpy(close, length=10)
    # Valid from index 10 (length=10, drift=1).
    valid = result[10:]
    assert ((valid >= 0.0) & (valid <= 100.0)).all()


@pytest.mark.momentum
def test_psl_all_up_all_down() -> None:
    up = np.arange(1.0, 26.0)
    down = np.arange(25.0, 0.0, -1.0)
    assert_allclose(psl_numpy(up, length=5)[5:], 100.0, rtol=1e-12)
    assert_allclose(psl_numpy(down, length=5)[5:], 0.0, rtol=1e-12)


@pytest.mark.momentum
def test_psl_zero_change_is_neutral() -> None:
    # diff == 0 counts neither up nor down.
    close = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = psl_numpy(close, length=4)
    # window diffs [2-2, 3-2, 3-3, 4-3] -> up count 2 of 4
    assert_allclose(result[4], 50.0, rtol=1e-12)


@pytest.mark.momentum
def test_psl_drift() -> None:
    close = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1])
    result = psl_numpy(close, length=3, drift=2)
    expected = _psl_reference(close, 3, 2)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_psl_nan_propagation() -> None:
    close = np.arange(1.0, 21.0)
    close[10] = np.nan
    result = psl_numpy(close, length=5)
    # diff[k] = close[k] - close[k-1] is NaN for k in {10, 11}; the
    # diff window [i-4, i] touches them for i in [10, 15].
    assert np.isnan(result[10:16]).all()
    assert not np.isnan(result[5:10]).any()
    assert not np.isnan(result[16:]).any()


@pytest.mark.momentum
def test_psl_kernel_bitwise_vs_numpy(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    assert_array_equal(_psl_numba(close, 8, 1), psl_numpy(close, length=8))


@pytest.mark.momentum
def test_psl_offset_fillna(prices_random_walk) -> None:
    base = psl_numpy(prices_random_walk, length=12)
    shifted = psl_numpy(prices_random_walk, length=12, offset=3, fillna=-1.0)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    expected = np.where(np.isnan(base), -1.0, base)
    assert_array_equal(shifted[:3], np.full(3, -1.0))
    assert_array_equal(shifted[3:], expected[:-3])


@pytest.mark.momentum
@pytest.mark.parametrize('length, drift', [(0, 1), (5, 0), (-2, 1)])
def test_psl_invalid_params(length: int, drift: int) -> None:
    with pytest.raises(ValueError):
        psl_numpy(np.arange(20.0), length=length, drift=drift)


@pytest.mark.momentum
def test_psl_empty_and_short() -> None:
    assert psl_numpy(np.array([], dtype=np.float64), length=4).size == 0
    assert np.isnan(
        psl_numpy(np.array([1.0, 2.0, 3.0]), length=5)
    ).all()


@pytest.mark.momentum
def test_psl_ind_numpy_and_series(prices_random_walk) -> None:
    expected = psl_numpy(prices_random_walk, length=7)
    from_array = psl_ind(prices_random_walk, length=7)
    from_series = psl_ind(pl.Series(prices_random_walk), length=7)
    assert_allclose(from_array, expected, rtol=1e-12, equal_nan=True)
    assert_allclose(from_series, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_psl_polars(df_random_walk: pl.DataFrame) -> None:
    expected = psl_numpy(df_random_walk['close'].to_numpy(), length=12)
    result = psl_polars(df_random_walk, length=12)
    assert 'PSL_12' in result.columns
    assert_allclose(
        result['PSL_12'].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )


@pytest.mark.momentum
def test_psl_polars_int_column_with_null() -> None:
    df = pl.DataFrame(
        {'close': [1, 2, None, 4, 5, 6, 7, 8, 9, 10]},
        schema={'close': pl.Int64},
    )
    result = psl_polars(df, length=4)
    assert result['PSL_4'].dtype == pl.Float64
    v = result['PSL_4'].to_numpy()
    assert np.isnan(v[:7]).all()  # windows touching the null at index 2
    assert not np.isnan(v[7:]).any()


@pytest.mark.momentum
def test_psl_readonly_input(prices_random_walk) -> None:
    arr = prices_random_walk.copy()
    arr.setflags(write=False)
    expected = psl_numpy(prices_random_walk, length=6)
    assert_allclose(
        psl_numpy(arr, length=6), expected, rtol=1e-12, equal_nan=True
    )
