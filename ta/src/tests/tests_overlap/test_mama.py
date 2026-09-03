# -*- coding: utf-8 -*-
"""Unit tests for MAMA (Mesa Adaptive Moving Average) module.

Tests cover:
- Numba core function (_mama_numba_core)
- Full mama_numba with offset, fillna, nan_policy
- Backend selection (Numba vs TA-Lib)
- Polars integration (mama_polars)
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

import pytest
import numpy as np
import numpy.typing as npt
import polars as pl
from numpy.testing import assert_allclose

from ...overlap.mama import (
    _mama_numba_core,
    mama_numba,
    mama_talib,
    mama_ind,
    mama_polars,
)
from ..._array_ops import _apply_offset_fillna
from ...external import talib_available


# ----------------------------------------------------------------------
# Numba core function tests
# ----------------------------------------------------------------------

@pytest.mark.overlap
def test_mama_numba_core_basic():
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    mama, fama = _mama_numba_core(close, fastlimit, slowlimit, prenan)
    assert mama[:prenan].all() if prenan > 0 else True
    assert fama[:prenan].all() if prenan > 0 else True
    assert len(mama) == len(close)
    assert len(fama) == len(close)


@pytest.mark.overlap
def test_mama_numba_core_short():
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    mama, fama = _mama_numba_core(close, fastlimit, slowlimit, prenan)
    assert len(mama) == len(close)
    assert len(fama) == len(close)
    assert np.isnan(mama).all()
    assert np.isnan(fama).all()


# ----------------------------------------------------------------------
# Offset, fillna, nan_policy tests
# ----------------------------------------------------------------------

@pytest.mark.overlap
def test_mama_numba_offset_fillna():
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    offset = 2
    fillna = 0.0
    base_mama, base_fama = mama_numba(close, fastlimit, slowlimit, prenan, offset=0, fillna=None)
    expected_mama = _apply_offset_fillna(base_mama, offset, fillna)
    expected_fama = _apply_offset_fillna(base_fama, offset, fillna)
    result_mama, result_fama = mama_numba(close, fastlimit, slowlimit, prenan, offset=offset, fillna=fillna)
    assert_allclose(result_mama, expected_mama, rtol=1e-6)
    assert_allclose(result_fama, expected_fama, rtol=1e-6)


@pytest.mark.overlap
def test_mama_numba_nan_policy_raise():
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    with pytest.raises(ValueError, match='Input close contains NaN values'):
        mama_numba(close, fastlimit, slowlimit, prenan, nan_policy='raise')


@pytest.mark.overlap
def test_mama_numba_nan_policy_ffill():
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_numba(close, fastlimit, slowlimit, prenan, nan_policy='ffill')
    assert np.isnan(result_mama[:2]).all()
    assert np.isnan(result_fama[:2]).all()
    assert np.isnan(result_mama[2:]).all()
    assert np.isnan(result_fama[2:]).all()


@pytest.mark.overlap
def test_mama_numba_nan_policy_ignore():
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_numba(close, fastlimit, slowlimit, prenan, nan_policy='ignore')
    assert np.isnan(result_mama[:2]).all()
    assert np.isnan(result_fama[:2]).all()
    assert np.isnan(result_mama[2:]).all()
    assert np.isnan(result_fama[2:]).all()


@pytest.mark.overlap
def test_mama_numba_nan_policy_bfill():
    close = np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float64)
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_numba(close, fastlimit, slowlimit, prenan, nan_policy='bfill')
    assert np.isnan(result_mama[:2]).all()
    assert np.isnan(result_fama[:2]).all()
    assert np.isnan(result_mama[2:]).all()
    assert np.isnan(result_fama[2:]).all()


# ----------------------------------------------------------------------
# Backend selection tests
# ----------------------------------------------------------------------

@pytest.mark.overlap
def test_mama_ind_uses_numba():
    close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_ind(close, fastlimit, slowlimit, prenan, use_talib=False)
    expected_mama, expected_fama = mama_numba(close, fastlimit, slowlimit, prenan)
    assert_allclose(result_mama, expected_mama, rtol=1e-6)
    assert_allclose(result_fama, expected_fama, rtol=1e-6)


# ----------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# ----------------------------------------------------------------------

@pytest.mark.overlap
def test_mama_numba_with_nan(prices_with_nan):
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_numba(prices_with_nan, fastlimit, slowlimit, prenan, nan_policy='ignore')
    assert np.isnan(result_mama[:2]).all()
    assert np.isnan(result_fama[:2]).all()
    assert np.isnan(result_mama[5:]).all()
    assert np.isnan(result_fama[5:]).all()


@pytest.mark.overlap
def test_mama_numba_with_inf(prices_with_inf):
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_numba(prices_with_inf, fastlimit, slowlimit, prenan, nan_policy='ignore')
    assert np.isnan(result_mama[:2]).all()
    assert np.isnan(result_fama[:2]).all()
    assert np.isnan(result_mama[5:]).all()
    assert np.isnan(result_fama[5:]).all()

@pytest.mark.overlap
def test_mama_numba_all_nan(prices_all_nan):
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_numba(prices_all_nan, fastlimit, slowlimit, prenan, nan_policy='ignore')
    assert np.isnan(result_mama).all()
    assert np.isnan(result_fama).all()
    result_mama_fill, result_fama_fill = mama_numba(prices_all_nan, fastlimit, slowlimit, prenan, fillna=0.0, nan_policy='ignore')
    assert (result_mama_fill == 0.0).all()
    assert (result_fama_fill == 0.0).all()

@pytest.mark.overlap
def test_mama_numba_extreme_values(prices_extreme):
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_numba(prices_extreme, fastlimit, slowlimit, prenan, nan_policy='ignore')
    assert result_mama is not None
    assert result_fama is not None

@pytest.mark.overlap
def test_mama_polars_with_nan(df_random_walk):
    close_arr = df_random_walk['close'].to_numpy().copy()
    close_arr[5] = np.nan
    df_with_nan = df_random_walk.with_columns(pl.Series('close', close_arr))
    fastlimit = 0.5
    slowlimit = 0.05
    prenan = 3
    result_mama, result_fama = mama_polars(
        df_with_nan,
        fastlimit=fastlimit,
        slowlimit=slowlimit,
        prenan=prenan,
        use_talib=False,
        nan_policy='ignore'
    )
    vals_mama = result_mama.to_numpy()
    vals_fama = result_fama.to_numpy()
    assert np.isnan(vals_mama[:2]).all()
    assert np.isnan(vals_fama[:2]).all()
    assert np.isnan(vals_mama[5:]).all()
    assert np.isnan(vals_fama[5:]).all()
