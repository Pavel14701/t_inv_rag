# -*- coding: utf-8 -*-
"""Offset/fillna contract tests shared by all statistics indicators.

Every module in ``ta.src.statistics`` finalises its output through
``_apply_offset_fillna`` (see ``ta/src/_array_ops.py``), which defines the
canonical semantics verified here at indicator level:

* positive ``offset`` shifts values forward; the first ``offset`` positions
  become ``fillna`` (or NaN when ``fillna`` is None);
* negative ``offset`` shifts backward; the last ``abs(offset)`` positions
  become ``fillna`` (or NaN);
* a given ``fillna`` also replaces every NaN of the shifted series,
  including the natural warm-up NaNs of a rolling indicator;
* ``offset=0`` with ``fillna`` only replaces the warm-up NaNs.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from ...statistics.entropy import entropy_numba
from ...statistics.mad import mad_numba
from ...statistics.median import median_numba
from ...statistics.quantile import quantile_numba
from ...statistics.skew import skew_numba
from ...statistics.stdev import stdev_numba
from ...statistics.variance import variance_numba
from ...statistics.zscore import zscore_numpy

LENGTH = 12
WARMUP = LENGTH - 1  # leading NaNs common to all rolling statistics here

STATISTICS_FNS: list[object] = [
    pytest.param(
        lambda c, **kw: entropy_numba(c, length=LENGTH, base=2.0, **kw),
        id='entropy',
    ),
    pytest.param(lambda c, **kw: mad_numba(c, length=LENGTH, **kw), id='mad'),
    pytest.param(
        lambda c, **kw: median_numba(c, length=LENGTH, **kw), id='median',
    ),
    pytest.param(
        lambda c, **kw: quantile_numba(c, length=LENGTH, q=0.5, **kw),
        id='quantile',
    ),
    pytest.param(
        lambda c, **kw: skew_numba(c, length=LENGTH, **kw), id='skew',
    ),
    pytest.param(
        lambda c, **kw: stdev_numba(c, length=LENGTH, ddof=1, **kw),
        id='stdev',
    ),
    pytest.param(
        lambda c, **kw: variance_numba(c, length=LENGTH, ddof=1, **kw),
        id='variance',
    ),
    pytest.param(
        lambda c, **kw: zscore_numpy(c, length=LENGTH, use_talib=False, **kw),
        id='zscore',
    ),
]


@pytest.mark.statistics
@pytest.mark.parametrize('fn', STATISTICS_FNS)
def test_offset_without_fillna_shifts_and_keeps_nan(
    fn: Callable[..., npt.NDArray[np.float64]],
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """offset=1 shifts values forward, NaNs are preserved when fillna=None."""
    close = prices_random_walk
    base = fn(close, offset=0)
    result = fn(close, offset=1)

    assert result.shape == close.shape
    assert np.isnan(result[0])
    # Warm-up NaNs (positions 0..WARMUP-1 shifted to 1..WARMUP) survive.
    assert np.isnan(result[1 : WARMUP + 1]).all()
    assert np.isfinite(result[WARMUP + 1 :]).all()
    assert_allclose(result[1:], base[:-1], rtol=1e-8, equal_nan=True)


@pytest.mark.statistics
@pytest.mark.parametrize('fn', STATISTICS_FNS)
def test_offset_with_fillna_replaces_warmup_nan(
    fn: Callable[..., npt.NDArray[np.float64]],
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """offset=1 + fillna shifts values AND fills all NaNs (incl. warm-up)."""
    close = prices_random_walk
    base = fn(close, offset=0)
    result = fn(close, offset=1, fillna=0.0)

    assert result[0] == 0.0
    expected = np.where(np.isnan(base[:-1]), 0.0, base[:-1])
    assert_allclose(result[1:], expected, rtol=1e-8)
    assert np.isfinite(result).all()


@pytest.mark.statistics
@pytest.mark.parametrize('fn', STATISTICS_FNS)
def test_negative_offset(
    fn: Callable[..., npt.NDArray[np.float64]],
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """offset=-1 shifts values backward; the tail becomes NaN."""
    close = prices_random_walk
    base = fn(close, offset=0)
    result = fn(close, offset=-1)

    assert np.isnan(result[-1])
    assert_allclose(result[:-1], base[1:], rtol=1e-8, equal_nan=True)


@pytest.mark.statistics
@pytest.mark.parametrize('fn', STATISTICS_FNS)
def test_zero_offset_fillna_only_replaces_nan(
    fn: Callable[..., npt.NDArray[np.float64]],
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """offset=0 + fillna must touch only the warm-up NaN positions."""
    close = prices_random_walk
    base = fn(close, offset=0)
    result = fn(close, offset=0, fillna=-1.0)

    expected = np.where(np.isnan(base), -1.0, base)
    assert_allclose(result, expected, rtol=1e-8)
    assert_allclose(result[WARMUP:], base[WARMUP:], rtol=1e-8)
