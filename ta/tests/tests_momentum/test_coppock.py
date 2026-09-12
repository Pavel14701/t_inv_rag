# -*- coding: utf-8 -*-
"""Unit tests for Coppock Curve module.

Covers:
- formula parity WMA(ROC(fast) + ROC(slow), wma_length)
- backend parity (numpy vs talib)
- warm-up NaNs end at slow + wma_length - 2
- regression: WMA no longer raises on the inherent ROC warm-up NaNs
- parameter validation
- offset / fillna semantics
- coppock_polars DataFrame integration
"""

import warnings

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.momentum.coppock import coppock_ind, coppock_numpy, coppock_polars
from ta.src.momentum.roc import roc_ind
from ta.src.overlap.wma import wma_ind


def _np(arr: object) -> np.ndarray:
    assert isinstance(arr, np.ndarray)
    return arr


@pytest.mark.momentum
def test_coppock_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Coppock equals WMA(ROC(fast) + ROC(slow), wma_length)."""
    close = prices_random_walk
    fast, slow, wma_length = 11, 14, 10
    result = coppock_numpy(
        close,
        fast=fast,
        slow=slow,
        wma_length=wma_length,
        use_talib=False,
    )
    total_roc = _np(roc_ind(close, length=fast, use_talib=False)) + _np(
        roc_ind(close, length=slow, use_talib=False)
    )
    expected = _np(
        wma_ind(
            total_roc,
            length=wma_length,
            use_talib=False,
            nan_policy="ignore",
        )
    )
    assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_coppock_backend_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy and talib backends agree on the finite tail."""
    common = dict(fast=11, slow=14, wma_length=10)
    a = coppock_numpy(prices_random_walk, **common, use_talib=False)
    b = coppock_numpy(prices_random_walk, **common, use_talib=True)
    assert_allclose(a[25:], b[25:], rtol=1e-9, equal_nan=True)


@pytest.mark.momentum
def test_coppock_warmup_nan_no_raise(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Regression: WMA must not raise on ROC warm-up NaNs.

    Previously coppock_numpy raised ValueError for every input because
    wma_ind defaulted to nan_policy='raise'.
    """
    slow, wma_length = 14, 10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = coppock_numpy(
            prices_random_walk,
            slow=slow,
            wma_length=wma_length,
            use_talib=False,
        )
    start = slow + wma_length - 1
    assert np.isnan(result[:start]).all()
    assert np.isfinite(result[start:]).all()


@pytest.mark.momentum
def test_coppock_non_contiguous(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided views give the same result."""
    expected = coppock_numpy(prices_random_walk, use_talib=False)
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    close_nc = mk_nc(prices_random_walk)
    assert not close_nc.flags.c_contiguous
    assert_allclose(
        coppock_numpy(close_nc, use_talib=False),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_coppock_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """fast/slow/wma_length < 1 are rejected."""
    with pytest.raises(ValueError, match="fast must be >= 1"):
        coppock_numpy(prices_random_walk, fast=0, use_talib=False)
    with pytest.raises(ValueError, match="slow must be >= 1"):
        coppock_numpy(prices_random_walk, slow=0, use_talib=False)
    with pytest.raises(ValueError, match="wma_length must be >= 1"):
        coppock_numpy(prices_random_walk, wma_length=0, use_talib=False)


@pytest.mark.momentum
def test_coppock_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    slow, wma_length = 14, 10
    r0 = coppock_numpy(
        prices_random_walk, slow=slow, wma_length=wma_length, use_talib=False
    )
    r2 = coppock_numpy(
        prices_random_walk,
        slow=slow,
        wma_length=wma_length,
        offset=2,
        fillna=0.0,
        use_talib=False,
    )
    assert r2[0] == 0.0 and r2[1] == 0.0
    start = slow + wma_length - 1  # first finite index of r0
    assert_allclose(
        r2[start + 2 :],
        r0[start:-2],
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_coppock_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """coppock_polars default column is COPC_{fast}_{slow}_{wma}."""
    result = coppock_polars(df_ohlc, use_talib=False)
    assert "COPC_11_14_10" in result.columns
    assert result["COPC_11_14_10"].dtype == pl.Float64
    expected = coppock_numpy(
        df_ohlc["close"].to_numpy(),
        use_talib=False,
    )
    assert_allclose(
        result["COPC_11_14_10"].to_numpy(), expected, equal_nan=True
    )


@pytest.mark.momentum
def test_coppock_ind_accepts_series(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """coppock_ind accepts numpy arrays and polars Series alike."""
    expected = coppock_numpy(prices_random_walk, use_talib=False)
    assert_allclose(
        coppock_ind(pl.Series(prices_random_walk), use_talib=False),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )
