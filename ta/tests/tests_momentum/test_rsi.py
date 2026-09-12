# -*- coding: utf-8 -*-
"""Unit tests for Relative Strength Index (RSI) module.

Tests cover:
- _compute_gain_loss_numba against numpy reference (exact), NaN propagation,
  empty input
- rsi_numpy: reference match, monotonic/constant/symmetric series, warm-up
  NaNs, drift, custom scalar, trim, offset, fillna
- TA-Lib cross-check (monotonic exact + convergent tail) and fallback for
  parameters TA-Lib does not support (drift>1, scalar!=100, length=1, NaN in)
- Parameter validation (length, drift, too-short, ndim, nan_policy)
- IEEE 754 compliance: NaN propagation, Inf->NaN (no input mutation),
  0/0 without warnings, overflow (1e308 diffs), all-NaN, empty, extreme values
- rsi_ind (ndarray / pl.Series / list / non-contiguous)
- rsi_polars DataFrame integration (default/custom col, nulls, NaN policy)
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ...external import talib_available
from ...momentum.rsi import (
    _compute_gain_loss_numba,
    rsi_ind,
    rsi_numpy,
    rsi_polars,
)
from ...overlap.rma import rma_ind


# -----------------------------------------------------------------------------
# Reference implementation (pure numpy orchestration on the same rma_ind,
# so the comparison is independent of the RMA seeding convention).
# Edge-rule order mirrors rsi_numpy: the 0/0 rule comes LAST and wins.
# -----------------------------------------------------------------------------
def _rsi_reference(
    close: npt.NDArray[np.float64],
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
) -> npt.NDArray[np.float64]:
    """Pure numpy RSI: manual diffs, rma_ind smoothing, explicit edge rules."""
    n = len(close)
    diff = np.zeros(n, dtype=np.float64)
    diff[drift:] = close[drift:] - close[: n - drift]
    gain = np.where(diff > 0.0, diff, 0.0)
    loss = np.where(diff < 0.0, -diff, 0.0)
    # IEEE 754: NaN diffs propagate (np.where's False branch would zero them)
    gain[np.isnan(diff)] = np.nan
    loss[np.isnan(diff)] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_gain = rma_ind(gain, length, nan_policy='ignore')
        avg_loss = rma_ind(loss, length, nan_policy='ignore')
        rs = avg_gain / avg_loss
        rsi = scalar - scalar / (1.0 + rs)
    rsi = np.where(avg_loss == 0.0, scalar, rsi)
    rsi = np.where(avg_gain == 0.0, 0.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss == 0.0), np.nan, rsi)
    rsi[np.isnan(gain)] = np.nan
    return rsi


# -----------------------------------------------------------------------------
# _compute_gain_loss_numba
# -----------------------------------------------------------------------------
@pytest.mark.momentum
def test_rsi_gain_loss_matches_numpy() -> None:
    """Numba gain/loss must be bitwise equal to the numpy reference."""
    close = np.array([1.0, 3.0, 2.0, 2.0, 5.0, 4.0, 4.0, 6.0, 7.0])
    for drift in (1, 2, 3):
        gain, loss = _compute_gain_loss_numba(close, drift)
        diff = np.zeros_like(close)
        diff[drift:] = close[drift:] - close[:-drift]
        assert_array_equal(gain, np.where(diff > 0, diff, 0.0))
        assert_array_equal(loss, np.where(diff < 0, -diff, 0.0))


@pytest.mark.momentum
def test_rsi_gain_loss_nan_propagation() -> None:
    """NaN prices must propagate: diff involving NaN yields NaN gain/loss.

    Regression test: the old implementation silently treated NaN diffs as
    'no change' (gain = loss = 0).
    """  # noqa: D403
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    gain, loss = _compute_gain_loss_numba(close, 1)
    assert np.isnan(gain[2]) and np.isnan(loss[2])   # nan - 2
    assert np.isnan(gain[3]) and np.isnan(loss[3])   # 4 - nan
    assert np.isfinite(gain[1]) and np.isfinite(loss[1])
    assert np.isfinite(gain[4]) and np.isfinite(loss[4])


@pytest.mark.momentum
def test_rsi_gain_loss_empty() -> None:
    """Empty input yields two empty arrays (no IndexError)."""
    gain, loss = _compute_gain_loss_numba(np.array([], dtype=np.float64), 1)
    assert gain.size == 0
    assert loss.size == 0


# -----------------------------------------------------------------------------
# rsi_numpy: correctness
# -----------------------------------------------------------------------------
@pytest.mark.momentum
def test_rsi_numpy_matches_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Native RSI path must match the pure numpy reference exactly."""
    for length in (2, 5, 14):
        result = rsi_numpy(prices_random_walk, length=length, use_talib=False)
        expected = _rsi_reference(prices_random_walk, length=length)
        assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_rsi_drift_matches_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Drift > 1 must use lagged diffs (native path)."""
    for drift in (2, 5):
        result = rsi_numpy(
            prices_random_walk, length=10, drift=drift, use_talib=False
        )
        expected = _rsi_reference(prices_random_walk, length=10, drift=drift)
        assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_rsi_monotonic_series() -> None:
    """Pure uptrend -> exactly 100, pure downtrend -> exactly 0."""
    up = np.arange(1.0, 41.0)
    down = up[::-1].copy()
    for length in (2, 14):
        rsi_up = rsi_numpy(up, length=length, use_talib=False)
        rsi_down = rsi_numpy(down, length=length, use_talib=False)
        # warm-up is NaN under either RMA seeding convention
        assert np.isnan(rsi_up[:length - 1]).all()
        assert np.isnan(rsi_down[:length - 1]).all()
        # from index `length` both conventions are settled
        assert (rsi_up[length:] == 100.0).all()
        assert (rsi_down[length:] == 0.0).all()


@pytest.mark.momentum
def test_rsi_constant_series_is_nan() -> None:
    """Flat market: 0/0 is undefined -> NaN everywhere (IEEE 754).

    Regression test for the edge-rule order: the both-zero -> NaN rule must
    be applied LAST, otherwise 'only losses -> 0' wins and flat RSI is 0.
    """
    close = np.full(40, 123.456)
    result = rsi_numpy(close, length=14, use_talib=False)
    assert np.isnan(result).all()


@pytest.mark.momentum
def test_rsi_bounded_0_100(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """All finite RSI values lie in [0, 100] regardless of the code path."""
    result = rsi_numpy(prices_random_walk, length=14)
    finite = result[np.isfinite(result)]
    assert finite.size > 0
    assert (finite >= 0.0).all()
    assert (finite <= 100.0).all()


@pytest.mark.momentum
def test_rsi_symmetric_series_near_50() -> None:
    """Equal gain/loss amplitudes drive RSI towards 50."""
    n = 120
    close = 100.0 + np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    result = rsi_numpy(close, length=14, use_talib=False)
    tail = result[5 * 14:]
    assert np.isfinite(tail).all()
    assert np.abs(tail - 50.0).max() < 5.0


@pytest.mark.momentum
def test_rsi_warmup_nans(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """First length-1 values are NaN (RMA warm-up), the rest are finite."""
    length = 14
    result = rsi_numpy(prices_random_walk, length=length, use_talib=False)
    assert np.isnan(result[:length - 1]).all()
    assert np.isfinite(result[length:]).all()
    assert len(result) == len(prices_random_walk)


@pytest.mark.momentum
def test_rsi_custom_scalar() -> None:
    """scalar=1.0 maps an uptrend to exactly 1.0 (native path honours it)."""
    up = np.arange(1.0, 41.0)
    result = rsi_numpy(up, length=5, scalar=1.0, use_talib=False)
    assert (result[5:] == 1.0).all()


@pytest.mark.momentum
def test_rsi_length_one_falls_back_to_native() -> None:
    """length=1 is valid but unsupported by TA-Lib -> native fallback.

    Regression test for the edge-rule order: the first (undiffed) bar has
    avg_gain == avg_loss == 0 and must resolve to NaN, not to 0.
    """
    close = np.arange(1.0, 21.0)
    result = rsi_numpy(close, length=1, use_talib=True)
    assert np.isnan(result[0])
    finite = result[np.isfinite(result)]
    # monotonic uptrend: every defined value is exactly 100
    assert (finite == 100.0).all()


# -----------------------------------------------------------------------------
# TA-Lib cross-check and fallback
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not talib_available, reason='TA-Lib is not installed')
@pytest.mark.momentum
def test_rsi_talib_matches_numpy_monotonic() -> None:
    """For monotonic series both paths give exactly 100 / 0."""
    up = np.arange(1.0, 61.0)
    down = up[::-1].copy()
    for close in (up, down):
        tal = rsi_numpy(close, length=14, use_talib=True)
        num = rsi_numpy(close, length=14, use_talib=False)
        assert_allclose(num[14:], tal[14:], rtol=1e-12, equal_nan=True)


@pytest.mark.skipif(not talib_available, reason='TA-Lib is not installed')
@pytest.mark.momentum
def test_rsi_talib_matches_numpy_tail() -> None:
    """After 20 warm-up cycles both Wilder recursions converge."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(400))
    tal = rsi_numpy(close, length=14, use_talib=True)
    num = rsi_numpy(close, length=14, use_talib=False)
    assert_allclose(num[20 * 14:], tal[20 * 14:], rtol=1e-3, atol=1e-3)


@pytest.mark.momentum
def test_rsi_talib_fallback_on_unsupported_params(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """drift>1 / scalar!=100 must bypass TA-Lib even if it is available.

    Regression test: the old implementation silently returned TA-Lib's
    drift=1 / scalar=100 result for these inputs.
    """
    for kwargs in ({'drift': 2}, {'scalar': 50.0}):
        with_talib = rsi_numpy(prices_random_walk, use_talib=True, **kwargs)
        pure = rsi_numpy(prices_random_walk, use_talib=False, **kwargs)
        assert_array_equal(with_talib, pure)


@pytest.mark.momentum
def test_rsi_talib_fallback_on_nan_input(prices_with_nan) -> None:
    """NaN input with nan_policy='ignore' must bypass TA-Lib: TA-Lib's NaN
    semantics are version-dependent, the native path enforces the contract.
    """  # noqa: D403
    with_talib = rsi_numpy(
        prices_with_nan, length=5, nan_policy='ignore', use_talib=True
    )
    pure = rsi_numpy(
        prices_with_nan, length=5, nan_policy='ignore', use_talib=False
    )
    assert_array_equal(with_talib, pure)
    assert np.isnan(with_talib[5:7]).all()


# -----------------------------------------------------------------------------
# Parameter validation
# -----------------------------------------------------------------------------
@pytest.mark.momentum
def test_rsi_invalid_length() -> None:  # noqa: D103
    close = np.arange(1.0, 21.0)
    for bad in (0, -5):
        with pytest.raises(ValueError, match='length must be >= 1'):
            rsi_numpy(close, length=bad)


@pytest.mark.momentum
def test_rsi_invalid_drift() -> None:  # noqa: D103
    close = np.arange(1.0, 21.0)
    with pytest.raises(ValueError, match='drift must be >= 1'):
        rsi_numpy(close, drift=0)


@pytest.mark.momentum
def test_rsi_too_short() -> None:
    """Fewer than length+drift values raise instead of silent NaN."""
    close = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='Input series too short'):
        rsi_numpy(close, length=5, drift=1)   # needs 6
    with pytest.raises(ValueError, match='Input series too short'):
        rsi_numpy(close, length=2, drift=3)   # needs 5


@pytest.mark.momentum
def test_rsi_empty(prices_empty) -> None:
    """Empty input raises ValueError (series too short)."""
    with pytest.raises(ValueError, match='Input series too short'):
        rsi_numpy(prices_empty, length=5)


@pytest.mark.momentum
def test_rsi_rejects_2d_input() -> None:  # noqa: D103
    with pytest.raises(ValueError, match='1-dimensional'):
        rsi_numpy(np.ones((10, 2)), length=5)


@pytest.mark.momentum
def test_rsi_invalid_nan_policy() -> None:  # noqa: D103
    close = np.arange(1.0, 21.0)
    with pytest.raises(ValueError, match='nan_policy'):
        rsi_numpy(close, length=5, nan_policy='drop')


# -----------------------------------------------------------------------------
# NaN / Inf handling (nan_policy) — IEEE 754
# -----------------------------------------------------------------------------
@pytest.mark.momentum
def test_rsi_nan_policy_raise(prices_with_nan) -> None:
    """Default nan_policy='raise' rejects input containing NaN."""
    with pytest.raises(ValueError, match='NaN'):
        rsi_numpy(prices_with_nan, length=5)


@pytest.mark.momentum
def test_rsi_nan_policy_ffill(prices_with_nan) -> None:
    """nan_policy='ffill' fills the input NaN and yields finite output."""
    result = rsi_numpy(
        prices_with_nan, length=5, nan_policy='ffill', use_talib=False
    )
    assert np.isnan(result[:4]).all()          # warm-up only
    assert np.isfinite(result[5:]).all()       # finite under either convention


@pytest.mark.momentum
def test_rsi_nan_policy_ignore_propagates_nan(prices_with_nan) -> None:
    """With 'ignore', diffs touching the NaN close must yield NaN RSI.

    NaN is at index 5 (see conftest); drift=1 -> diffs at 5 and 6 are
    undefined, so RSI must be NaN there (native path enforces the mask).
    """
    result = rsi_numpy(
        prices_with_nan, length=5, nan_policy='ignore', use_talib=False
    )
    assert len(result) == len(prices_with_nan)
    assert not np.isinf(result).any()
    assert np.isnan(result[5:7]).all()


@pytest.mark.momentum
def test_rsi_inf_replaced_with_nan(prices_with_inf) -> None:
    """Inf is converted to NaN and handled
    by nan_policy (never an Inf error).
    """
    with pytest.raises(ValueError, match='NaN'):
        rsi_numpy(prices_with_inf, length=5)   # default policy: NaN -> raise
    result = rsi_numpy(
        prices_with_inf, length=5, nan_policy='ignore', use_talib=False
    )
    assert not np.isinf(result).any()
    assert len(result) == len(prices_with_inf)


@pytest.mark.momentum
def test_rsi_input_not_mutated(prices_with_inf) -> None:
    """Inf->NaN conversion must work on a copy, never on the caller's array."""
    snapshot = prices_with_inf.copy()
    with pytest.raises(ValueError):
        rsi_numpy(prices_with_inf, length=5)
    assert_array_equal(prices_with_inf, snapshot)


@pytest.mark.momentum
def test_rsi_all_nan(prices_all_nan) -> None:
    """All-NaN input: raise by default, all-NaN with 'ignore', fillna works."""
    with pytest.raises(ValueError, match='NaN'):
        rsi_numpy(prices_all_nan, length=5)
    result = rsi_numpy(
        prices_all_nan, length=5, nan_policy='ignore', use_talib=False
    )
    assert np.isnan(result).all()
    filled = rsi_numpy(
        prices_all_nan, length=5, fillna=-1.0, nan_policy='ignore',
        use_talib=False,
    )
    assert (filled == -1.0).all()


@pytest.mark.momentum
def test_rsi_no_floating_point_warnings(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """0/0 (flat stretch) and x/0 must be silenced by np.errstate."""
    close = prices_random_walk.copy()
    close[10:20] = close[11]   # exact flat stretch -> zero diffs
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        result = rsi_numpy(
            close, length=5,
            nan_policy='ignore', use_talib=False
        )
    assert len(result) == len(close)


@pytest.mark.momentum
def test_rsi_overflow_produces_no_inf() -> None:
    """Differences overflowing to +/-Inf must not crash and never leak Inf."""
    close = np.array([
        1e308, -1e308, 1e308, -1e308,
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ])
    result = rsi_numpy(close, length=5, nan_policy='ignore', use_talib=False)
    assert len(result) == len(close)
    assert not np.isinf(result).any()
    finite = result[np.isfinite(result)]
    assert ((finite >= 0.0) & (finite <= 100.0)).all()


@pytest.mark.momentum
def test_rsi_extreme_values(prices_extreme) -> None:
    """Extreme values (1e300, 1e-300) must not crash; output stays bounded."""
    result = rsi_numpy(
        prices_extreme, length=5, nan_policy='ignore', use_talib=False
    )
    assert len(result) == len(prices_extreme)
    assert not np.isinf(result).any()
    finite = result[np.isfinite(result)]
    assert ((finite >= 0.0) & (finite <= 100.0)).all()


# -----------------------------------------------------------------------------
# trim / offset / fillna
# -----------------------------------------------------------------------------
@pytest.mark.momentum
def test_rsi_trim(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """trim=True removes exactly length-1 warm-up values."""
    length = 10
    full = rsi_numpy(prices_random_walk, length=length, use_talib=False)
    trimmed = rsi_numpy(
        prices_random_walk, length=length, use_talib=False, trim=True
    )
    assert len(trimmed) == len(prices_random_walk) - length + 1
    assert_array_equal(trimmed, full[length - 1:])


@pytest.mark.momentum
def test_rsi_offset_positive(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Positive offset shifts result forward, filling the head with NaN."""
    base = rsi_numpy(prices_random_walk, length=10, use_talib=False)
    shifted = rsi_numpy(
        prices_random_walk, length=10,
        use_talib=False, offset=3
    )
    assert np.isnan(shifted[:3]).all()
    assert_allclose(shifted[3:], base[:-3], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_rsi_offset_negative(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Negative offset shifts result backward, filling the tail with NaN."""
    base = rsi_numpy(prices_random_walk, length=10, use_talib=False)
    shifted = rsi_numpy(
        prices_random_walk, length=10,
        use_talib=False, offset=-3
    )
    assert np.isnan(shifted[-3:]).all()
    assert_allclose(shifted[:-3], base[3:], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_rsi_trim_then_offset(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset is applied after trimming."""
    length = 10
    trimmed = rsi_numpy(
        prices_random_walk, length=length, use_talib=False, trim=True
    )
    both = rsi_numpy(
        prices_random_walk, length=length, use_talib=False, trim=True, offset=2
    )
    assert np.isnan(both[:2]).all()
    assert_allclose(both[2:], trimmed[:-2], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_rsi_fillna(prices_random_walk: npt.NDArray[np.float64]) -> None:
    """Fillna replaces warm-up NaNs."""
    result = rsi_numpy(
        prices_random_walk, length=10,
        use_talib=False, fillna=0.0
    )
    assert not np.isnan(result).any()
    assert (result[:9] == 0.0).all()


# -----------------------------------------------------------------------------
# rsi_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.momentum
def test_rsi_ind_with_ndarray(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    assert_array_equal(
        rsi_ind(prices_random_walk, length=14),
        rsi_numpy(prices_random_walk, length=14),
    )


@pytest.mark.momentum
def test_rsi_ind_with_pl_series(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    s = pl.Series(prices_random_walk)
    result = rsi_ind(s, length=14)
    expected = rsi_numpy(prices_random_walk, length=14)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_rsi_ind_accepts_list() -> None:  # noqa: D103
    result = rsi_ind([float(i) for i in range(1, 31)], length=5)
    expected = rsi_numpy(np.arange(1.0, 31.0), length=5)
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_rsi_ind_non_contiguous() -> None:
    """Strided (non-contiguous) view gives the same values as contiguous."""
    close = np.arange(1.0, 41.0)
    interleaved = np.empty((40, 2))
    interleaved[:, 0] = close
    interleaved[:, 1] = close
    view = interleaved[:, 0]
    assert not view.flags.c_contiguous
    assert_allclose(
        rsi_ind(view, length=5, use_talib=False),
        rsi_ind(close, length=5, use_talib=False),
        rtol=1e-12, equal_nan=True,
    )


# -----------------------------------------------------------------------------
# rsi_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.momentum
def test_rsi_polars_basic(df_random_walk: pl.DataFrame) -> None:  # noqa: D103, E501
    result_df = rsi_polars(df_random_walk, length=14, output_col='RSI')
    assert 'RSI' in result_df.columns
    assert result_df['RSI'].dtype == pl.Float64
    assert len(result_df) == len(df_random_walk)
    expected = rsi_numpy(df_random_walk['close'].to_numpy(), length=14)
    assert_allclose(
        result_df['RSI'].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )


@pytest.mark.momentum
def test_rsi_polars_default_output_col() -> None:  # noqa: D103
    df = pl.DataFrame({'close': np.arange(1.0, 31.0)})
    assert 'RSI_5' in rsi_polars(df, length=5).columns


@pytest.mark.momentum
def test_rsi_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:  # noqa: D103, E501
    expected = rsi_numpy(
        df_random_walk['close'].to_numpy(), length=10, offset=2, fillna=0.0
    )
    result_df = rsi_polars(
        df_random_walk, length=10, offset=2, fillna=0.0, output_col='RSI'
    )
    assert_allclose(
        result_df['RSI'].to_numpy(), expected, rtol=1e-12, equal_nan=True
    )


@pytest.mark.momentum
def test_rsi_polars_with_nan(df_random_walk: pl.DataFrame) -> None:
    """rsi_polars propagates NaN with nan_policy='ignore'."""
    close = df_random_walk['close'].to_numpy().copy()
    close[5] = np.nan
    df_nan = df_random_walk.with_columns(pl.Series('close', close))
    result_df = rsi_polars(
        df_nan, length=5, output_col='RSI', nan_policy='ignore'
    )
    vals = result_df['RSI'].to_numpy()
    assert np.isnan(vals[5:7]).all()
    assert not np.isinf(vals).any()
    assert len(result_df) == len(df_nan)


@pytest.mark.momentum
def test_rsi_polars_float_column_with_null() -> None:
    """Polars nulls become NaN and are handled by nan_policy."""
    vals = [float(i) for i in range(1, 31)]
    vals[5] = None
    df = pl.DataFrame({'close': vals})
    result_df = rsi_polars(df, length=5, nan_policy='ignore', output_col='RSI')
    v = result_df['RSI'].to_numpy()
    assert np.isnan(v[5:7]).all()
    assert not np.isinf(v).any()


@pytest.mark.momentum
def test_rsi_polars_int_column_with_null() -> None:
    """Int64 column with nulls is cast to Float64 instead of failing."""
    df = pl.DataFrame(
        {'close': [1, 2, None, 4, 5, 6, 7, 8, 9, 10]},
        schema={'close': pl.Int64},
    )
    result_df = rsi_polars(df, length=5, nan_policy='ignore', output_col='RSI')
    assert result_df['RSI'].dtype == pl.Float64
    v = result_df['RSI'].to_numpy()
    assert np.isnan(v[2:4]).all()   # diffs at 2 and 3 touch the null at 2


# -----------------------------------------------------------------------------
# Read-only input buffers (IEEE 754 / robustness)
# -----------------------------------------------------------------------------
@pytest.mark.momentum
def test_rsi_readonly_input(prices_with_nan) -> None:
    """Read-only arrays must not crash numba kernels.

    Regression test: polars' zero-copy to_numpy() returns a readonly view,
    and numba's mutable float64[:] signature rejects it with
    'TypeError: No matching definition for argument type(s)
    readonly array(float64, 1d, C)'. The library must normalize the input.
    """
    # 1) NaN + 'ignore' -> native path with a readonly buffer
    arr = prices_with_nan.copy()
    arr.setflags(write=False)
    assert not arr.flags.writeable
    result = rsi_numpy(arr, length=5, nan_policy='ignore', use_talib=False)
    assert np.isnan(result[5:7]).all()          # NaN propagation intact
    assert len(result) == len(arr)

    # 2) clean data -> TA-Lib path with a readonly buffer
    clean = np.arange(1.0, 21.0)
    clean.setflags(write=False)
    result_talib = rsi_numpy(clean, length=5)
    assert len(result_talib) == len(clean)

    # 3) readonly AND non-contiguous at once
    base = np.arange(1.0, 81.0)
    view = base[::2]
    assert not view.flags.c_contiguous
    view.setflags(write=False)
    expected = rsi_numpy(
        np.ascontiguousarray(view), length=5, use_talib=False
    )
    assert_allclose(
        rsi_numpy(view, length=5, use_talib=False),
        expected, rtol=1e-12, equal_nan=True,
    )