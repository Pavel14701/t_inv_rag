# -*- coding: utf-8 -*-
"""Unit tests for HiLo Activator indicator.

Tests cover:
- Core Numba logic (_hilo_numba_core)
- _hilo_numba against reference (using the same MA function)
- _hilo_talib (if TA-Lib available)
- offset and fillna
- hilo_ind with Polars Series
- hilo_polars DataFrame integration
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.hilo import (
    _hilo_numba_core,
    _hilo_numba,
    _hilo_talib,
    hilo_ind,
    hilo_polars,
    ma_numba,
)
from ..._array_ops import _apply_offset_fillna
from ...external import talib_available


# -----------------------------------------------------------------------------
# Reference implementation using the same MA function as the code
# -----------------------------------------------------------------------------
def _hilo_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    high_length: int,
    low_length: int,
    mamode: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:  # noqa: E501
    """Pure Python reference HiLo Activator (uses ma_numba for consistency)."""
    n = len(close)
    hilo = np.full(n, np.nan, dtype=np.float64)
    long_arr = np.full(n, np.nan, dtype=np.float64)
    short_arr = np.full(n, np.nan, dtype=np.float64)
    high_ma = ma_numba(high, high_length, mamode, nan_policy='ignore')
    low_ma = ma_numba(low, low_length, mamode, nan_policy='ignore')
    if n < 2:
        return hilo, long_arr, short_arr
    for i in range(1, n):
        if close[i] > high_ma[i - 1]:
            hilo[i] = low_ma[i]
            long_arr[i] = low_ma[i]
            short_arr[i] = np.nan
        elif close[i] < low_ma[i - 1]:
            hilo[i] = high_ma[i]
            short_arr[i] = high_ma[i]
            long_arr[i] = np.nan
        else:
            hilo[i] = hilo[i - 1]
            long_arr[i] = hilo[i - 1]
            short_arr[i] = hilo[i - 1]
    return hilo, long_arr, short_arr


# -----------------------------------------------------------------------------
# Tests for _hilo_numba_core (core logic)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hilo_core_against_reference(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    close = prices_random_walk + np.random.randn(n) * 0.2
    high_ma = ma_numba(high, 5, 'sma', nan_policy='ignore')
    low_ma = ma_numba(low, 10, 'sma', nan_policy='ignore')
    hilo_nb, long_nb, short_nb = _hilo_numba_core(
        high, low, close, high_ma, low_ma
    )
    n = len(close)
    hilo_ref = np.full(n, np.nan, dtype=np.float64)
    long_ref = np.full(n, np.nan, dtype=np.float64)
    short_ref = np.full(n, np.nan, dtype=np.float64)
    if n >= 2:
        for i in range(1, n):
            if close[i] > high_ma[i - 1]:
                hilo_ref[i] = low_ma[i]
                long_ref[i] = low_ma[i]
                short_ref[i] = np.nan
            elif close[i] < low_ma[i - 1]:
                hilo_ref[i] = high_ma[i]
                short_ref[i] = high_ma[i]
                long_ref[i] = np.nan
            else:
                hilo_ref[i] = hilo_ref[i - 1]
                long_ref[i] = hilo_ref[i - 1]
                short_ref[i] = hilo_ref[i - 1]
    assert_allclose(hilo_nb, hilo_ref, rtol=1e-6, equal_nan=True)
    assert_allclose(long_nb, long_ref, rtol=1e-6, equal_nan=True)
    assert_allclose(short_nb, short_ref, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for _hilo_numba (full Numba version)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hilo_numba_against_reference(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    close = prices_random_walk + np.random.randn(n) * 0.2
    for mamode in ('sma', 'ema'):
        hilo_nb, long_nb, short_nb = _hilo_numba(
            high, low, close,
            high_length=5,
            low_length=10,
            mamode=mamode,
            offset=0,
            fillna=None,
        )
        hilo_ref, long_ref, short_ref = _hilo_reference(
            high, low, close,
            high_length=5,
            low_length=10,
            mamode=mamode,
        )
        assert_allclose(hilo_nb, hilo_ref, rtol=1e-6, equal_nan=True)
        assert_allclose(long_nb, long_ref, rtol=1e-6, equal_nan=True)
        assert_allclose(short_nb, short_ref, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hilo_numba_offset_fillna(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    close = prices_random_walk + np.random.randn(n) * 0.2
    offset = 3
    fillna = 0.0
    hilo_base, long_base, short_base = _hilo_numba(
        high, low, close,
        high_length=5,
        low_length=10,
        mamode='sma',
        offset=0,
        fillna=None,
    )
    expected_hilo = _apply_offset_fillna(hilo_base, offset, fillna)
    expected_long = _apply_offset_fillna(long_base, offset, fillna)
    expected_short = _apply_offset_fillna(short_base, offset, fillna)
    hilo_act, long_act, short_act = _hilo_numba(
        high, low, close,
        high_length=5,
        low_length=10,
        mamode='sma',
        offset=offset,
        fillna=fillna,
    )
    assert_allclose(hilo_act, expected_hilo, rtol=1e-6, equal_nan=True)
    assert_allclose(long_act, expected_long, rtol=1e-6, equal_nan=True)
    assert_allclose(short_act, expected_short, rtol=1e-6, equal_nan=True)


@pytest.mark.overlap
def test_hilo_numba_short_window() -> None:  # noqa: D103
    high = np.array([1.0, 2.0], dtype=np.float64)
    low = np.array([0.5, 1.5], dtype=np.float64)
    close = np.array([1.2, 1.8], dtype=np.float64)
    hilo, long, short = _hilo_numba(
        high, low, close,
        high_length=5,
        low_length=5,
        mamode='sma',
    )
    assert np.isnan(hilo).all()
    assert np.isnan(long).all()
    assert np.isnan(short).all()


# -----------------------------------------------------------------------------
# Tests for _hilo_talib (if TA-Lib available)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(
    not talib_available,
    reason='TA-Lib not installed'
)
@pytest.mark.overlap
def test_hilo_talib_against_reference(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    close = prices_random_walk + np.random.randn(n) * 0.2
    for mamode in ('sma', 'ema'):
        hilo_tl, long_tl, short_tl = _hilo_talib(
            high, low, close,
            high_length=5,
            low_length=10,
            mamode=mamode,
        )
        hilo_ref, long_ref, short_ref = _hilo_reference(
            high, low, close,
            high_length=5,
            low_length=10,
            mamode=mamode,
        )
        # TA-Lib may have slightly different MA values, allow larger tolerance
        assert_allclose(
            hilo_tl, hilo_ref, rtol=1e-3,
            atol=1e-3, equal_nan=True
        )
        assert_allclose(
            long_tl, long_ref, rtol=1e-3,
            atol=1e-3, equal_nan=True
        )
        assert_allclose(
            short_tl, short_ref, rtol=1e-3,
            atol=1e-3, equal_nan=True
        )


@pytest.mark.skipif(
    not talib_available,
    reason='TA-Lib not installed'
)
@pytest.mark.overlap
def test_hilo_talib_offset_fillna(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    np.random.seed(42)
    n = len(prices_random_walk)
    high = prices_random_walk + np.random.randn(n) * 0.5
    low = prices_random_walk - np.random.randn(n) * 0.5
    close = prices_random_walk + np.random.randn(n) * 0.2
    offset = 3
    fillna = 0.0
    hilo_base, long_base, short_base = _hilo_talib(
        high, low, close,
        high_length=5,
        low_length=10,
        mamode='sma',
        offset=0,
        fillna=None,
    )
    expected_hilo = _apply_offset_fillna(hilo_base, offset, fillna)
    expected_long = _apply_offset_fillna(long_base, offset, fillna)
    expected_short = _apply_offset_fillna(short_base, offset, fillna)
    hilo_act, long_act, short_act = _hilo_talib(
        high, low, close,
        high_length=5,
        low_length=10,
        mamode='sma',
        offset=offset,
        fillna=fillna,
    )

    assert_allclose(hilo_act, expected_hilo, rtol=1e-6, equal_nan=True)
    assert_allclose(long_act, expected_long, rtol=1e-6, equal_nan=True)
    assert_allclose(short_act, expected_short, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for hilo_ind (universal wrapper)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hilo_ind_with_pl_series(  # noqa: D103
    prices_random_walk: npt.NDArray[np.float64]
) -> None:
    np.random.seed(42)
    n = len(prices_random_walk)
    high_s = pl.Series(prices_random_walk + np.random.randn(n) * 0.5)
    low_s = pl.Series(prices_random_walk - np.random.randn(n) * 0.5)
    close_s = pl.Series(prices_random_walk + np.random.randn(n) * 0.2)

    hilo, long, short = hilo_ind(
        high_s, low_s, close_s,
        high_length=5,
        low_length=10,
        mamode='sma',
        use_talib=False,
    )

    assert isinstance(hilo, np.ndarray)
    assert hilo.shape == (n,)
    assert hilo.dtype == np.float64

    high_np = high_s.to_numpy()
    low_np = low_s.to_numpy()
    close_np = close_s.to_numpy()
    hilo_ref, long_ref, short_ref = _hilo_reference(
        high_np, low_np, close_np,
        high_length=5,
        low_length=10,
        mamode='sma',
    )
    assert_allclose(hilo, hilo_ref, rtol=1e-6, equal_nan=True)
    assert_allclose(long, long_ref, rtol=1e-6, equal_nan=True)
    assert_allclose(short, short_ref, rtol=1e-6, equal_nan=True)


# -----------------------------------------------------------------------------
# Tests for hilo_polars (DataFrame integration)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hilo_polars_basic(df_random_walk: pl.DataFrame) -> None:  # noqa: D103
    np.random.seed(42)
    n = len(df_random_walk)
    df = df_random_walk.with_columns([
        pl.Series(
            'high',
            df_random_walk['close'].to_numpy() + np.abs(
                np.random.randn(n) * 0.5
            )
        ),
        pl.Series(
            'low', df_random_walk['close'].to_numpy() - np.abs(
                np.random.randn(n) * 0.5)
        ),
    ])

    high_length = 5
    low_length = 10
    result_df = hilo_polars(
        df,
        high_col='high',
        low_col='low',
        close_col='close',
        high_length=high_length,
        low_length=low_length,
        mamode='sma',
        use_talib=False,
        suffix='_test',
    )

    expected_cols = ['HILO_test', 'HILOl_test', 'HILOs_test']
    for col in expected_cols:
        assert col in result_df.columns
        assert result_df[col].dtype == pl.Float64
    assert len(result_df) == len(df)

    high_arr = df['high'].to_numpy()
    low_arr = df['low'].to_numpy()
    close_arr = df['close'].to_numpy()
    hilo_np, long_np, short_np = _hilo_numba(
        high_arr, low_arr, close_arr,
        high_length=high_length,
        low_length=low_length,
        mamode='sma',
    )
    assert_allclose(
        result_df['HILO_test'].to_numpy(),
        hilo_np, rtol=1e-6, equal_nan=True
    )
    assert_allclose(
        result_df['HILOl_test'].to_numpy(), long_np,
        rtol=1e-6, equal_nan=True
    )
    assert_allclose(
        result_df['HILOs_test'].to_numpy(), short_np,
        rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_hilo_polars_offset_fillna(df_random_walk: pl.DataFrame) -> None:  # noqa: D103, E501
    np.random.seed(42)
    n = len(df_random_walk)
    df = df_random_walk.with_columns([
        pl.Series(
            'high', df_random_walk['close'].to_numpy() + np.abs(
                np.random.randn(n) * 0.5
            )
        ),
        pl.Series(
            'low', df_random_walk['close'].to_numpy() - np.abs(
                np.random.randn(n) * 0.5
            )
        ),
    ])

    offset = 2
    fillna = 0.0
    result_df = hilo_polars(
        df,
        high_col='high',
        low_col='low',
        close_col='close',
        high_length=5,
        low_length=10,
        mamode='sma',
        offset=offset,
        fillna=fillna,
        suffix='_test',
    )

    high_arr = df['high'].to_numpy()
    low_arr = df['low'].to_numpy()
    close_arr = df['close'].to_numpy()
    hilo_np, long_np, short_np = _hilo_numba(
        high_arr, low_arr, close_arr,
        high_length=5,
        low_length=10,
        mamode='sma',
        offset=offset,
        fillna=fillna,
    )
    assert_allclose(
        result_df['HILO_test'].to_numpy(),
        hilo_np, rtol=1e-6, equal_nan=True
    )
    assert_allclose(
        result_df['HILOl_test'].to_numpy(),
        long_np, rtol=1e-6, equal_nan=True
    )
    assert_allclose(
        result_df['HILOs_test'].to_numpy(),
        short_np, rtol=1e-6, equal_nan=True
    )


@pytest.mark.overlap
def test_hilo_polars_ema_mode(df_random_walk: pl.DataFrame) -> None:  # noqa: D103, E501
    np.random.seed(42)
    n = len(df_random_walk)
    df = df_random_walk.with_columns([
        pl.Series(
            'high',
            df_random_walk['close'].to_numpy() + np.abs(
                np.random.randn(n) * 0.5
            )),
        pl.Series(
            'low',
            df_random_walk['close'].to_numpy() - np.abs(
                np.random.randn(n) * 0.5
            )),
    ])
    result_df = hilo_polars(
        df,
        high_col='high',
        low_col='low',
        close_col='close',
        high_length=5,
        low_length=10,
        mamode='ema',
        use_talib=False,
        suffix='_ema',
    )
    assert 'HILO_ema' in result_df.columns
    high_arr = df['high'].to_numpy()
    low_arr = df['low'].to_numpy()
    close_arr = df['close'].to_numpy()
    hilo_np, long_np, short_np = _hilo_numba(
        high_arr, low_arr, close_arr,
        high_length=5,
        low_length=10,
        mamode='ema',
    )
    assert_allclose(
        result_df['HILO_ema'].to_numpy(),
        hilo_np, rtol=1e-6, equal_nan=True
    )


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_hilo_numba_with_nan(prices_with_nan):
    """HiLo with NaN in input should propagate NaNs correctly."""  # noqa: D403
    high = prices_with_nan + 1.0
    low = prices_with_nan - 1.0
    close = prices_with_nan
    hilo, long_, short_ = _hilo_numba(
        high, low, close,
        high_length=3,
        low_length=5,
        mamode='sma',
    )
    # At least some NaNs should appear
    assert np.isnan(hilo).any()


@pytest.mark.overlap
def test_hilo_numba_with_inf(prices_with_inf):
    """Inf in input is replaced with NaN, so it behaves like NaN."""
    high = prices_with_inf + 1.0
    low = prices_with_inf - 1.0
    close = prices_with_inf
    hilo, long_, short_ = _hilo_numba(
        high, low, close,
        high_length=3,
        low_length=5,
        mamode='sma',
    )
    # Should not crash
    assert hilo is not None


@pytest.mark.overlap
def test_hilo_numba_empty(prices_empty):
    """Empty input should return empty arrays."""
    hilo, long_, short_ = _hilo_numba(
        prices_empty, prices_empty, prices_empty,
        high_length=3,
        low_length=5,
    )
    assert hilo.size == 0
    assert long_.size == 0
    assert short_.size == 0


@pytest.mark.overlap
def test_hilo_numba_all_nan(prices_all_nan):
    """All NaNs -> all NaNs (or fillna if provided)."""
    high = prices_all_nan
    low = prices_all_nan
    close = prices_all_nan
    hilo, long_, short_ = _hilo_numba(
        high, low, close,
        high_length=3,
        low_length=5,
        fillna=0.0,
    )
    # All values become fillna because _apply_offset_fillna replaces NaNs
    assert (hilo == 0.0).all()
    assert (long_ == 0.0).all()
    assert (short_ == 0.0).all()


@pytest.mark.overlap
def test_hilo_numba_extreme_values(prices_extreme):
    """Extreme values (1e300, 1e-300) must not crash."""
    high = prices_extreme
    low = prices_extreme
    close = prices_extreme
    hilo, long_, short_ = _hilo_numba(
        high, low, close,
        high_length=3,
        low_length=5,
    )
    assert hilo is not None


@pytest.mark.overlap
def test_hilo_polars_with_nan(df_random_walk):
    """Polars integration should handle NaN correctly."""
    np.random.seed(42)
    n = len(df_random_walk)
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns([
        pl.Series('close', close_arr),
        pl.Series('high', close_arr + np.abs(np.random.randn(n) * 0.5)),
        pl.Series('low', close_arr - np.abs(np.random.randn(n) * 0.5)),
    ])
    result_df = hilo_polars(
        df_with_nan,
        high_col='high',
        low_col='low',
        close_col='close',
        high_length=3,
        low_length=5,
        use_talib=False,
        suffix='_test',
    )
    assert 'HILO_test' in result_df.columns
    assert len(result_df) == len(df_random_walk)
    hilo_vals = result_df['HILO_test'].to_numpy()
    # At least some NaNs should appear
    assert np.isnan(hilo_vals).any()
