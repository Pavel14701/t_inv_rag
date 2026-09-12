# -*- coding: utf-8 -*-
"""Unit tests for rolling entropy (ENTP) module."""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.statistics.entropy import (
    entropy_ind,
    entropy_numba,
    entropy_polars,
)


@pytest.mark.statistics
def test_entropy_numba_basic(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test entropy_numba on a random walk."""
    close = prices_random_walk
    length = 10
    result = entropy_numba(close, length=length, base=2.0)

    assert result.shape == close.shape
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


@pytest.mark.statistics
def test_entropy_numba_constant() -> None:
    """Test entropy of a constant series (should be 0)."""
    prices = np.full(20, 5.0)
    length = 5
    result = entropy_numba(prices, length=length, base=2.0)
    # After the first valid window, all should be 0
    assert_allclose(result[length - 1 :], 0.0, rtol=1e-6)


@pytest.mark.statistics
def test_entropy_numba_uniform() -> None:
    """Test entropy of uniform distribution (max entropy for given length)."""
    # Uniform prices: 1,2,3,4,5
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    length = 5
    result = entropy_numba(prices, length=length, base=2.0)
    # Entropy of uniform 5 classes = log2(5) ~= 2.3219
    expected = np.log2(5)
    assert_allclose(result[-1], expected, rtol=1e-6)


@pytest.mark.statistics
def test_entropy_numba_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test offset and fillna."""
    close = prices_random_walk
    length = 10
    result_no_offset = entropy_numba(close, length=length, base=2.0, offset=0)
    result_offset = entropy_numba(
        close, length=length, base=2.0, offset=1, fillna=0.0
    )

    assert result_offset[0] == 0.0
    # fillna also replaces the warm-up NaNs of the shifted series
    no_offset_tail = result_no_offset[:-1]
    expected = np.where(np.isnan(no_offset_tail), 0.0, no_offset_tail)
    assert_allclose(result_offset[1:], expected, rtol=1e-6)


@pytest.mark.statistics
def test_entropy_numba_nonfinite_window_is_nan() -> None:
    """NaN/inf in a window force NaN there; later windows recover."""
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = entropy_numba(close, length=3)
    # Windows [0..2], [1..3], [2..4] contain the NaN at index 2.
    assert np.isnan(result[2:5]).all()
    assert np.isfinite(result[5:]).all()
    assert_allclose(result[5:], np.log2(3.0), rtol=1e-6)

    close_inf = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0])
    result_inf = entropy_numba(close_inf, length=3)
    assert np.isnan(result_inf[2:4]).all()
    assert np.isfinite(result_inf[4:]).all()


@pytest.mark.statistics
def test_entropy_numba_invalid_base_raises() -> None:
    """Base <= 0 or base == 1 is invalid (log base zero or negative)."""
    prices = np.array([1.0, 2.0, 3.0, 4.0])
    for bad_base in (0.0, -2.0, 1.0):
        with pytest.raises(ValueError, match="base must be positive"):
            entropy_numba(prices, length=2, base=bad_base)


@pytest.mark.statistics
def test_entropy_numba_length_too_short_raises() -> None:
    """Length < 2 is rejected (window of one point has no entropy)."""
    prices = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="length must be >= 2"):
        entropy_numba(prices, length=1)
    with pytest.raises(ValueError, match="length must be >= 2"):
        entropy_numba(prices, length=0)


@pytest.mark.statistics
def test_entropy_numba_length_exceeds_data() -> None:
    """If length > len(close), all outputs are NaN."""
    result = entropy_numba(np.array([1.0, 2.0, 3.0]), length=5)
    assert result.shape == (3,)
    assert np.isnan(result).all()


@pytest.mark.statistics
def test_entropy_numba_base_scaling() -> None:
    """base=e (nats) equals base=2 (bits) times ln(2)."""
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    length = 4
    bits = entropy_numba(prices, length=length, base=2.0)
    nats = entropy_numba(prices, length=length, base=np.e)
    assert_allclose(nats[length - 1 :], bits[length - 1 :] * np.log(2.0))


@pytest.mark.statistics
def test_entropy_ind_with_pl_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Test entropy_ind with Polars Series input."""
    s = pl.Series(prices_random_walk)
    length = 10
    result = entropy_ind(s, length=length, base=2.0)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(prices_random_walk),)
    assert result.dtype == np.float64
    assert np.isnan(result[: length - 1]).all()
    assert np.isfinite(result[length - 1 :]).all()


@pytest.mark.statistics
def test_entropy_polars_basic(df_random_walk: pl.DataFrame) -> None:
    """Test entropy_polars adds a column correctly."""
    length = 10
    result_series = entropy_polars(
        df_random_walk,
        close_col="close",
        length=length,
        base=2.0,
        output_col="ENTP",
    )

    assert isinstance(result_series, pl.Series)
    assert result_series.name == "ENTP"
    assert len(result_series) == len(df_random_walk)
    assert result_series.dtype == pl.Float64

    close_arr = df_random_walk["close"].to_numpy()
    expected = entropy_numba(close_arr, length=length, base=2.0)
    assert_allclose(
        result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True
    )


@pytest.mark.statistics
def test_entropy_polars_default_output_col() -> None:
    """Test default output column name."""
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
    length = 3
    result_series = entropy_polars(
        df, close_col="close", length=length, base=2.0
    )
    assert result_series.name == f"ENTP_{length}"


@pytest.mark.statistics
def test_entropy_polars_with_offset_fillna(
    df_random_walk: pl.DataFrame,
) -> None:
    """Test entropy_polars with offset and fillna."""
    length = 10
    result_series = entropy_polars(
        df_random_walk,
        close_col="close",
        length=length,
        base=2.0,
        offset=1,
        fillna=0.0,
        output_col="ENTP",
    )

    assert result_series[0] == 0.0

    close_arr = df_random_walk["close"].to_numpy()
    expected = entropy_numba(
        close_arr, length=length, base=2.0, offset=1, fillna=0.0
    )
    assert_allclose(
        result_series.to_numpy(), expected, rtol=1e-6, equal_nan=True
    )
