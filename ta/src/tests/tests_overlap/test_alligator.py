# -*- coding: utf-8 -*-
"""Unit tests for Alligator indicator (SMMA-based).

Tests cover:
- alligator_ind with sequential and parallel modes
- offset and fillna (sequential mode)
- Polars integration (alligator_polars)
- Consistency between parallel and sequential modes (without offset/fillna)
- IEEE 754 compliance: NaN, Inf, empty, short series
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.alligator import alligator_ind, alligator_polars


# -----------------------------------------------------------------------------
# Local reference SMMA implementation (pure Python)
# -----------------------------------------------------------------------------

def _smma_reference(
    close: npt.NDArray[np.float64],
    length: int
) -> npt.NDArray[np.float64]:
    """Pure Python reference SMMA."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    s = 0.0
    for i in range(length):
        s += close[i]
    out[length - 1] = s / length
    for i in range(length, n):
        out[i] = ((length - 1) * out[i - 1] + close[i]) / length
    return out


def _alligator_reference(close, jaw_len, teeth_len, lips_len):
    jaw = _smma_reference(close, jaw_len)
    teeth = _smma_reference(close, teeth_len)
    lips = _smma_reference(close, lips_len)
    return jaw, teeth, lips


def _apply_offset_fillna_test(
    arr: np.ndarray,
    offset: int,
    fillna: float
) -> np.ndarray:
    """Apply shift and fillna exactly as in the real implementation."""
    out = np.full_like(arr, np.nan)
    if offset > 0:
        out[offset:] = arr[:-offset]
    elif offset < 0:
        out[:offset] = arr[-offset:]
    else:
        out[:] = arr
    out[np.isnan(out)] = fillna
    return out


# -----------------------------------------------------------------------------
# Tests for alligator_ind
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_alligator_ind_sequential_vs_parallel(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    close = prices_random_walk
    jaw, teeth, lips = 13, 8, 5
    jaw_seq, teeth_seq, lips_seq = alligator_ind(
        close, jaw=jaw, teeth=teeth, lips=lips, parallel=False
    )
    jaw_par, teeth_par, lips_par = alligator_ind(
        close, jaw=jaw, teeth=teeth, lips=lips, parallel=True
    )
    assert_allclose(jaw_seq, jaw_par, rtol=1e-6, equal_nan=True)
    assert_allclose(teeth_seq, teeth_par, rtol=1e-6, equal_nan=True)
    assert_allclose(lips_seq, lips_par, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_alligator_ind_against_reference(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    close = prices_random_walk
    jaw, teeth, lips = 5, 3, 2
    jaw_actual, teeth_actual, lips_actual = alligator_ind(
        close, jaw=jaw, teeth=teeth, lips=lips, parallel=False
    )
    jaw_exp, teeth_exp, lips_exp = _alligator_reference(
        close, jaw, teeth, lips
    )
    assert_allclose(jaw_actual, jaw_exp, rtol=1e-6, equal_nan=True)
    assert_allclose(teeth_actual, teeth_exp, rtol=1e-6, equal_nan=True)
    assert_allclose(lips_actual, lips_exp, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_alligator_ind_parallel_against_reference(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    close = prices_random_walk
    jaw, teeth, lips = 7, 4, 3
    jaw_actual, teeth_actual, lips_actual = alligator_ind(
        close, jaw=jaw, teeth=teeth, lips=lips, parallel=True
    )
    jaw_exp, teeth_exp, lips_exp = _alligator_reference(
        close, jaw, teeth, lips
    )
    assert_allclose(jaw_actual, jaw_exp, rtol=1e-6, equal_nan=True)
    assert_allclose(teeth_actual, teeth_exp, rtol=1e-6, equal_nan=True)
    assert_allclose(lips_actual, lips_exp, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_alligator_ind_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    """Test offset and fillna using sequential mode."""
    close = prices_random_walk
    jaw, teeth, lips = 5, 3, 2
    offset = 2
    fillna = 0.0
    # Base results without offset/fillna (sequential)
    jaw_base, teeth_base, lips_base = alligator_ind(
        close, jaw=jaw, teeth=teeth, lips=lips,
        parallel=False, offset=0, fillna=None
    )
    # Manually apply offset with fillna
    expected_jaw = _apply_offset_fillna_test(jaw_base, offset, fillna)
    expected_teeth = _apply_offset_fillna_test(teeth_base, offset, fillna)
    expected_lips = _apply_offset_fillna_test(lips_base, offset, fillna)
    # Get results with offset/fillna from sequential mode
    jaw_seq, teeth_seq, lips_seq = alligator_ind(
        close, jaw=jaw, teeth=teeth, lips=lips,
        parallel=False, offset=offset, fillna=fillna
    )
    assert_allclose(jaw_seq, expected_jaw, rtol=1e-6, equal_nan=True)
    assert_allclose(teeth_seq, expected_teeth, rtol=1e-6, equal_nan=True)
    assert_allclose(lips_seq, expected_lips, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_alligator_ind_with_pl_series(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    s = pl.Series(prices_random_walk)
    jaw, teeth, lips = 5, 3, 2
    jaw_arr, teeth_arr, lips_arr = alligator_ind(
        s, jaw=jaw, teeth=teeth, lips=lips
    )
    assert isinstance(jaw_arr, np.ndarray)
    assert jaw_arr.shape == (len(prices_random_walk),)
    assert jaw_arr.dtype == np.float64
    assert np.isnan(jaw_arr[:jaw - 1]).all()
    assert np.isfinite(jaw_arr[jaw - 1:]).all()


# -----------------------------------------------------------------------------
# Tests for alligator_polars
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_alligator_polars_basic(  # noqa: D103
    df_random_walk: pl.DataFrame
) -> None:
    jaw, teeth, lips = 5, 3, 2
    result_df = alligator_polars(
        df_random_walk, close_col='close',
        jaw=jaw, teeth=teeth, lips=lips
    )
    expected_cols = [
        f'AGj_{jaw}_{teeth}_{lips}',
        f'AGt_{jaw}_{teeth}_{lips}',
        f'AGl_{jaw}_{teeth}_{lips}'
    ]
    for col in expected_cols:
        assert col in result_df.columns
        assert result_df[col].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    close_arr = df_random_walk['close'].to_numpy()
    jaw_np, teeth_np, lips_np = alligator_ind(
        close_arr, jaw=jaw, teeth=teeth, lips=lips
    )
    assert_allclose(
        result_df[expected_cols[0]].to_numpy(),
        jaw_np, rtol=1e-6, equal_nan=True
    )
    assert_allclose(
        result_df[expected_cols[1]].to_numpy(),
        teeth_np, rtol=1e-6, equal_nan=True
    )
    assert_allclose(
        result_df[expected_cols[2]].to_numpy(),
        lips_np, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_alligator_polars_with_suffix(df_random_walk: pl.DataFrame) -> None:  # noqa: D103, E501
    jaw, teeth, lips = 5, 3, 2
    suffix = '_custom'
    result_df = alligator_polars(
        df_random_walk, close_col='close', jaw=jaw,
        teeth=teeth, lips=lips, suffix=suffix
    )
    expected_cols = ['AGj_custom', 'AGt_custom', 'AGl_custom']
    for col in expected_cols:
        assert col in result_df.columns
    default_suffix = f'_{jaw}_{teeth}_{lips}'
    assert f'AGj{default_suffix}' not in result_df.columns


@pytest.mark.overlap
def test_alligator_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:  # noqa: D103, E501
    jaw, teeth, lips = 5, 3, 2
    offset = 2
    fillna = 0.0
    result_df = alligator_polars(
        df_random_walk,
        close_col='close',
        jaw=jaw,
        teeth=teeth,
        lips=lips,
        offset=offset,
        fillna=fillna,
    )
    close_arr = df_random_walk['close'].to_numpy()
    jaw_np, teeth_np, lips_np = alligator_ind(
        close_arr, jaw=jaw, teeth=teeth, lips=lips,
        offset=offset, fillna=fillna, parallel=True
    )
    col = f'AGj_{jaw}_{teeth}_{lips}'
    assert_allclose(
        result_df[col].to_numpy(),
        jaw_np, rtol=1e-6, equal_nan=True
    )
    col = f'AGt_{jaw}_{teeth}_{lips}'
    assert_allclose(
        result_df[col].to_numpy(),
        teeth_np, rtol=1e-6, equal_nan=True
    )
    col = f'AGl_{jaw}_{teeth}_{lips}'
    assert_allclose(
        result_df[col].to_numpy(),
        lips_np, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests
# -----------------------------------------------------------------------------

@pytest.mark.overlap
def test_alligator_ind_empty() -> None:
    """Empty input should return empty arrays without errors."""
    empty = np.array([])
    jaw, teeth, lips = alligator_ind(empty, parallel=True)
    assert jaw.size == 0
    assert teeth.size == 0
    assert lips.size == 0
    # Sequential mode also
    jaw, teeth, lips = alligator_ind(empty, parallel=False)
    assert jaw.size == 0
    assert teeth.size == 0
    assert lips.size == 0


@pytest.mark.overlap
def test_alligator_ind_short_series() -> None:
    """Series shorter than all periods -> all NaNs."""
    short = np.array([1.0, 2.0, 3.0])  # length 3
    jaw, teeth, lips = alligator_ind(short, parallel=True)
    assert np.isnan(jaw).all()
    assert np.isnan(teeth).all()
    assert np.isnan(lips).all()
    jaw, teeth, lips = alligator_ind(short, parallel=False)
    assert np.isnan(jaw).all()
    assert np.isnan(teeth).all()
    assert np.isnan(lips).all()


@pytest.mark.overlap
def test_alligator_ind_nan_input() -> None:
    """NaN in input propagates correctly (IEEE 754)."""  # noqa: D403
    data = np.array(
        [
            1.0, 2.0, np.nan, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0
        ],
        dtype=np.float64
    )
    jaw, teeth, lips = alligator_ind(
        data, parallel=True, nan_policy='ignore'
    )
    # Since SMMA is recursive, a single NaN will make subsequent values NaN
    # until enough new data 'washes' it out (but SMMA uses all previous).
    # Actually, SMMA uses previous value, so once NaN appears,
    # it stays NaN forever.
    # Check that at least at position 2 (where NaN is) it's NaN.
    assert np.isnan(jaw[2])
    assert np.isnan(teeth[2])
    assert np.isnan(lips[2])
    # And all later positions are also NaN (because of recursion)
    assert np.isnan(jaw[2:]).all()
    assert np.isnan(teeth[2:]).all()
    assert np.isnan(lips[2:]).all()
    # Sequential mode should behave the same
    jaw_seq, teeth_seq, lips_seq = alligator_ind(
        data, parallel=False, nan_policy='ignore'
    )
    assert np.isnan(jaw_seq[2:]).all()
    assert np.isnan(teeth_seq[2:]).all()
    assert np.isnan(lips_seq[2:]).all()


@pytest.mark.overlap
def test_alligator_ind_inf_input() -> None:
    """Inf in input is replaced with NaN (IEEE 754)."""
    data = np.array(
        [
            1.0, 2.0, np.inf, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0
        ],
        dtype=np.float64
    )
    jaw, teeth, lips = alligator_ind(
        data, parallel=True, nan_policy='ignore'
    )
    # Inf should be replaced by NaN inside the function,
    # so behaviour same as NaN input
    assert np.isnan(jaw[2:]).all()
    assert np.isnan(teeth[2:]).all()
    assert np.isnan(lips[2:]).all()


@pytest.mark.overlap
def test_alligator_ind_extreme_values() -> None:
    """Extreme values (1e300, 1e-300) must not crash."""
    extreme = np.array(
        [1e300, 1e-300, 1.0] * 10,
        dtype=np.float64
    )  # 30 elements
    jaw, teeth, lips = alligator_ind(extreme, parallel=True)
    # Should not crash; may contain inf/nan, but that's allowed
    assert jaw is not None
    assert teeth is not None
    assert lips is not None
    jaw, teeth, lips = alligator_ind(extreme, parallel=False)
    assert jaw is not None
    assert teeth is not None
    assert lips is not None


@pytest.mark.overlap
def test_alligator_ind_nan_policy_raise() -> None:
    """Input with NaN and default nan_policy='raise' raises ValueError."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match='NaN'):
        alligator_ind(data, jaw=3, teeth=2, lips=2)


@pytest.mark.overlap
def test_alligator_ind_nan_policy_ffill() -> None:
    """Input with NaN and nan_policy='ffill' is filled and computed."""
    data = np.array(
        [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        dtype=np.float64,
    )
    jaw, teeth, lips = alligator_ind(
        data, jaw=3, teeth=2, lips=2, nan_policy='ffill'
    )
    # after warmup all values must be finite
    assert np.isfinite(jaw[2:]).all()
    assert np.isfinite(teeth[1:]).all()
    assert np.isfinite(lips[1:]).all()


@pytest.mark.overlap
def test_alligator_ind_invalid_periods() -> None:
    """Periods below 1 raise ValueError."""
    close = np.linspace(1.0, 20.0, 20)
    with pytest.raises(ValueError, match='must all be >= 1'):
        alligator_ind(close, jaw=0, teeth=2, lips=2)
    with pytest.raises(ValueError, match='must all be >= 1'):
        alligator_ind(close, jaw=3, teeth=-1, lips=2)


@pytest.mark.overlap
def test_alligator_ind_negative_prices() -> None:
    """Negative prices should not break the calculation."""
    neg = np.linspace(-10, 0, 50)
    jaw, teeth, lips = alligator_ind(neg, parallel=True)
    # Should be finite after the initial NaN period
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    assert np.isfinite(jaw[jaw_period - 1:]).all()
    assert np.isfinite(teeth[teeth_period - 1:]).all()
    assert np.isfinite(lips[lips_period - 1:]).all()
