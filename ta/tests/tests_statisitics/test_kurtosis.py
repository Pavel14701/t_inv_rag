# -*- coding: utf-8 -*-
"""Unit tests for rolling excess kurtosis (KURT) module."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.statistics.kurtosis import (
    kurtosis_ind,
    kurtosis_numba,
    kurtosis_polars,
)


@pytest.mark.statistics
def test_kurtosis_numba_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test kurtosis_numba on a random walk."""
    close = prices_random_walk
    length = 30
    result = kurtosis_numba(close, length=length)

    assert result.shape == close.shape
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


@pytest.mark.statistics
def test_kurtosis_numba_known_value() -> None:
    """Test the documented example: windows of [1,2,3,4] give -1.2."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = kurtosis_numba(prices, length=4)
    assert_allclose(result[3:], -1.2, rtol=1e-6)


@pytest.mark.statistics
def test_kurtosis_numba_scipy_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test parity with scipy's bias-corrected kurtosis (bias=False)."""
    from scipy.stats import kurtosis as sp_kurtosis

    close = prices_random_walk
    length = 30
    result = kurtosis_numba(close, length=length)

    for i in range(length - 1, len(close), 7):
        expected = sp_kurtosis(close[i - length + 1 : i + 1], bias=False)
        assert_allclose(result[i], expected, rtol=1e-8, atol=1e-10)


@pytest.mark.statistics
def test_kurtosis_numba_constant() -> None:
    """Constant windows have zero variance -> NaN (undefined kurtosis)."""
    prices = np.full(20, 5.0)
    result = kurtosis_numba(prices, length=5)
    assert np.isnan(result).all()


@pytest.mark.statistics
def test_kurtosis_numba_unbiased_for_normal() -> None:
    """E[kurtosis] of normal samples is ~0 (bias-corrected estimator)."""
    rng = np.random.default_rng(42)
    length = 40
    values = []
    for _ in range(500):
        sample = rng.normal(0.0, 1.0, length)
        r = kurtosis_numba(sample, length=length)
        if np.isfinite(r[-1]):
            values.append(r[-1])
    assert abs(np.mean(values)) < 0.5


@pytest.mark.statistics
def test_kurtosis_numba_large_prices_accuracy() -> None:
    """Two-pass form stays accurate where running power sums cancel away."""
    from scipy.stats import kurtosis as sp_kurtosis

    rng = np.random.default_rng(0)
    close = 100000.0 + rng.normal(0.0, 0.01, 120)
    length = 20
    result = kurtosis_numba(close, length=length)

    for i in range(length - 1, len(close), 9):
        expected = sp_kurtosis(close[i - length + 1 : i + 1], bias=False)
        assert_allclose(result[i], expected, rtol=1e-6, atol=1e-8)


@pytest.mark.statistics
def test_kurtosis_numba_nan_recovers() -> None:
    """NaN poisons only the windows that contain it, then output recovers."""
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = kurtosis_numba(close, length=4)
    # Windows [0..3], [1..4], [2..5] all contain the NaN at index 2.
    assert np.isnan(result[3:6]).all()
    # Windows from index 6 on no longer contain it.
    assert np.isfinite(result[6:]).all()
    assert_allclose(result[6:], -1.2, rtol=1e-6)


@pytest.mark.statistics
def test_kurtosis_numba_inf_is_nan_and_recovers() -> None:
    """A +-inf value makes its windows NaN, later windows recover."""
    close = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = kurtosis_numba(close, length=4)
    # Windows [0..3] and [1..4] contain the inf at index 1.
    assert np.isnan(result[3:5]).all()
    assert np.isfinite(result[5:]).all()


@pytest.mark.statistics
def test_kurtosis_numba_length_too_short_raises() -> None:
    """Kurtosis needs length >= 4 (estimator denominator is zero below)."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="length must be >= 4"):
        kurtosis_numba(prices, length=3)


@pytest.mark.statistics
def test_kurtosis_numba_length_exceeds_data() -> None:
    """If length > len(close), all outputs are NaN."""
    result = kurtosis_numba(np.array([1.0, 2.0, 3.0]), length=4)
    assert result.shape == (3,)
    assert np.isnan(result).all()


@pytest.mark.statistics
def test_kurtosis_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 30
    result_no_offset = kurtosis_numba(close, length=length, offset=0)
    result_offset = kurtosis_numba(close, length=length, offset=1, fillna=0.0)

    assert result_offset[0] == 0.0
    # fillna also replaces the warm-up NaNs of the shifted series
    no_offset_tail = result_no_offset[:-1]
    expected = np.where(np.isnan(no_offset_tail), 0.0, no_offset_tail)
    assert_allclose(result_offset[1:], expected, rtol=1e-6)


@pytest.mark.statistics
def test_kurtosis_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test kurtosis_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 30
    result = kurtosis_ind(s, length=length)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(prices_random_walk),)
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


@pytest.mark.statistics
def test_kurtosis_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test kurtosis_polars adds a column correctly."""
    length = 30
    result_df = kurtosis_polars(
        df_random_walk,
        close_col="close",
        length=length,
        output_col="KURT",
    )

    assert "KURT" in result_df.columns
    assert len(result_df) == len(df_random_walk)

    close_arr = df_random_walk["close"].to_numpy()
    expected = kurtosis_numba(close_arr, length=length)
    assert_allclose(
        result_df["KURT"].to_numpy(),
        expected,
        rtol=1e-6,
        equal_nan=True,
    )


@pytest.mark.statistics
def test_kurtosis_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    result_df = kurtosis_polars(df, close_col="close", length=4)
    assert "KURT_4" in result_df.columns
