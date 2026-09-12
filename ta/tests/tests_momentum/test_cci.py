# -*- coding: utf-8 -*-
"""Unit tests for Commodity Channel Index (CCI) module.

Covers:
- formula parity (TP - SMA(TP)) / (c * MAD(TP))
- backend parity (numpy vs talib)
- regression: fillna must not leak into the hlc3/SMA/MAD inputs
- warm-up NaNs (length - 1)
- non-contiguous inputs (regression: no-op contiguous loop)
- parameter validation (length)
- offset / fillna semantics
- cci_polars DataFrame integration
"""

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src.momentum.cci import cci_ind, cci_numpy, cci_polars
from ta.src.overlap.hlc3 import hlc3_ind
from ta.src.overlap.sma import sma_ind
from ta.src.statistics.mad import mad_ind


def _np(arr: object) -> np.ndarray:
    assert isinstance(arr, np.ndarray)
    return arr


def _hl_arrays(
    prices_random_walk: npt.NDArray[np.float64],
    seed: int = 41,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    noise = np.abs(rng.normal(0.0, 0.5, len(prices_random_walk))) + 0.1
    return prices_random_walk + noise, prices_random_walk - noise


@pytest.mark.momentum
def test_cci_matches_formula(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy backend equals (TP - SMA(TP)) / (c * MAD(TP))."""
    high, low = _hl_arrays(prices_random_walk)
    length, c = 14, 0.015
    result = cci_numpy(
        high, low, prices_random_walk, length=length, c=c, use_talib=False
    )
    tp = _np(hlc3_ind(high, low, prices_random_walk))
    mean_tp = _np(sma_ind(tp, length=length, use_talib=False))
    mad_tp = _np(mad_ind(tp, length=length))
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (tp - mean_tp) / (c * mad_tp)
    assert_allclose(result, expected, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_cci_backend_parity(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Numpy and talib backends agree on the finite tail."""
    high, low = _hl_arrays(prices_random_walk)
    a = cci_numpy(high, low, prices_random_walk, length=14, use_talib=False)
    b = cci_numpy(high, low, prices_random_walk, length=14, use_talib=True)
    assert_allclose(a[13:], b[13:], rtol=1e-8, equal_nan=True)


@pytest.mark.momentum
def test_cci_fillna_does_not_leak(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Regression: fillna must only affect the output, not the inputs.

    Previously fillna was forwarded to hlc3_ind, changing the SMA/MAD
    inputs and therefore every tail value.
    """
    high, low = _hl_arrays(prices_random_walk)
    base = cci_numpy(high, low, prices_random_walk, length=14, use_talib=False)
    filled = cci_numpy(
        high, low, prices_random_walk, length=14, fillna=0.0, use_talib=False
    )
    assert np.isnan(base[:13]).all()
    assert filled[0] == 0.0 and not np.isnan(filled).any()
    assert_allclose(filled[13:], base[13:], rtol=1e-12)


@pytest.mark.momentum
def test_cci_warmup_nan(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN for the first length - 1 positions, finite after."""
    high, low = _hl_arrays(prices_random_walk)
    result = cci_numpy(
        high, low, prices_random_walk, length=14, use_talib=False
    )
    assert np.isnan(result[:13]).all()
    assert np.isfinite(result[13:]).all()


@pytest.mark.momentum
def test_cci_non_contiguous(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Strided views give the same result (no-op contiguous regression)."""
    high, low = _hl_arrays(prices_random_walk)
    expected = cci_numpy(high, low, prices_random_walk, use_talib=False)
    mk_nc = lambda a: np.stack([a, a], axis=1)[:, 0]  # noqa: E731
    high_nc, low_nc, close_nc = (
        mk_nc(high),
        mk_nc(low),
        mk_nc(prices_random_walk),
    )
    assert not high_nc.flags.c_contiguous
    assert_allclose(
        cci_numpy(high_nc, low_nc, close_nc, use_talib=False),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        cci_ind(
            pl.Series(high),
            pl.Series(low),
            pl.Series(prices_random_walk),
            use_talib=False,
        ),
        expected,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_cci_validation_raises(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Length < 1 is rejected."""
    high, low = _hl_arrays(prices_random_walk)
    with pytest.raises(ValueError, match="length must be >= 1"):
        cci_numpy(high, low, prices_random_walk, length=0, use_talib=False)


@pytest.mark.momentum
def test_cci_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset shifts and fillna replaces ALL NaN (incl. warm-up)."""
    high, low = _hl_arrays(prices_random_walk)
    length = 14
    r0 = cci_numpy(
        high, low, prices_random_walk, length=length, use_talib=False
    )
    r2 = cci_numpy(
        high,
        low,
        prices_random_walk,
        length=length,
        offset=2,
        fillna=0.0,
        use_talib=False,
    )
    assert r2[0] == 0.0 and r2[1] == 0.0
    assert_allclose(
        r2[length + 1 :],
        r0[length - 1 : -2],
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_cci_polars_basic(df_ohlc: pl.DataFrame) -> None:
    """cci_polars default column is CCI_{length}_{c}."""
    result = cci_polars(df_ohlc, use_talib=False)
    assert "CCI_14_0.015" in result.columns
    assert result["CCI_14_0.015"].dtype == pl.Float64
    expected = cci_numpy(
        df_ohlc["high"].to_numpy(),
        df_ohlc["low"].to_numpy(),
        df_ohlc["close"].to_numpy(),
        use_talib=False,
    )
    assert_allclose(
        result["CCI_14_0.015"].to_numpy(), expected, equal_nan=True
    )
