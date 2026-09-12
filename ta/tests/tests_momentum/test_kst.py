# -*- coding: utf-8 -*-
"""Unit tests for Know Sure Thing (KST)."""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_array_equal

from ta.src.momentum.kst import kst_ind, kst_numpy, kst_polars
from ta.src.momentum.roc import roc_ind
from ta.src.overlap.sma import sma_ind


def _kst_reference(
    close: np.ndarray,
    rocs=(10, 15, 20, 30),
    smas=(10, 10, 10, 15),
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure numpy KST reference."""
    kst = np.zeros_like(close)
    for i, (roc_len, sma_len) in enumerate(zip(rocs, smas)):
        roc = roc_ind(close, length=roc_len, use_talib=False)
        smoothed = sma_ind(
            roc, length=sma_len, use_talib=False, nan_policy="ignore"
        )
        kst = kst + (i + 1.0) * smoothed
    signalma = sma_ind(
        kst, length=signal, use_talib=False, nan_policy="ignore"
    )
    return kst, signalma


@pytest.mark.momentum
def test_kst_matches_reference(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    expected_kst, expected_signal = _kst_reference(close)
    kst, signalma = kst_numpy(close)
    assert_allclose(kst, expected_kst, rtol=1e-10, equal_nan=True)
    assert_allclose(signalma, expected_signal, rtol=1e-10, equal_nan=True)


@pytest.mark.momentum
def test_kst_weighted_sum_identity(prices_random_walk) -> None:
    # kst == s1 + 2*s2 + 3*s3 + 4*s4 for the component smoothings.
    close = np.ascontiguousarray(prices_random_walk)
    kst, _ = kst_numpy(close)
    total = np.zeros_like(close)
    for weight, roc_len, sma_len in (
        (1.0, 10, 10),
        (2.0, 15, 10),
        (3.0, 20, 10),
        (4.0, 30, 15),
    ):
        roc = roc_ind(close, length=roc_len, use_talib=False)
        smoothed = sma_ind(
            roc, length=sma_len, use_talib=False, nan_policy="ignore"
        )
        total += weight * smoothed
    assert_allclose(kst, total, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_kst_warmup_nan(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    kst, signalma = kst_numpy(close)
    # ROC(30) valid from 30; its SMA(15) valid from 30 + 14 = 44.
    assert np.isnan(kst[:44]).all()
    assert not np.isnan(kst[44:]).any()
    # Signal SMA(9) of kst valid from 44 + 8 = 52.
    assert np.isnan(signalma[:52]).all()


@pytest.mark.momentum
def test_kst_signal_is_sma_of_kst(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    kst, signalma = kst_numpy(close)
    expected = sma_ind(kst, length=9, use_talib=False, nan_policy="ignore")
    assert_allclose(signalma, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_kst_custom_periods() -> None:
    close = np.linspace(10.0, 40.0, 80)
    kst, _ = kst_numpy(
        close,
        roc1=3,
        roc2=4,
        roc3=5,
        roc4=6,
        sma1=2,
        sma2=2,
        sma3=2,
        sma4=2,
        signal=2,
    )
    # Longest chain: ROC(6) valid from 6, SMA(2) from 7.
    assert np.isnan(kst[:7]).all()
    assert not np.isnan(kst[7:]).any()


@pytest.mark.momentum
@pytest.mark.parametrize(
    "kw",
    [
        {"roc1": 0},
        {"roc4": -1},
        {"sma2": 0},
        {"signal": -3},
    ],
)
def test_kst_invalid_params(kw: dict) -> None:
    with pytest.raises(ValueError):
        kst_numpy(np.arange(60.0), **kw)


@pytest.mark.momentum
def test_kst_empty_input() -> None:
    kst, signalma = kst_numpy(np.array([], dtype=np.float64))
    assert kst.size == 0
    assert signalma.size == 0


@pytest.mark.momentum
def test_kst_offset_fillna(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    base_kst, base_signal = kst_numpy(close)
    kst, signalma = kst_numpy(close, offset=2, fillna=0.0)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    exp_kst = np.where(np.isnan(base_kst), 0.0, base_kst)
    exp_signal = np.where(np.isnan(base_signal), 0.0, base_signal)
    assert_array_equal(kst[:2], np.zeros(2))
    assert_allclose(kst[2:], exp_kst[:-2], rtol=1e-12)
    assert_allclose(signalma[2:], exp_signal[:-2], rtol=1e-12)


@pytest.mark.momentum
def test_kst_ind_numpy_and_series(prices_random_walk) -> None:
    close = np.ascontiguousarray(prices_random_walk)
    expected = kst_numpy(close)
    from_arrays = kst_ind(close)
    from_series = kst_ind(pl.Series(close))
    for res, exp in zip(from_arrays, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
    for res, exp in zip(from_series, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_kst_polars(df_random_walk: pl.DataFrame) -> None:
    close = df_random_walk["close"].to_numpy()
    kst, signalma = kst_numpy(close)
    result = kst_polars(df_random_walk)
    assert "KST_10_15_20_30" in result.columns
    assert "KSTs_10_15_20_30" in result.columns
    assert_allclose(
        result["KST_10_15_20_30"].to_numpy(),
        kst,
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        result["KSTs_10_15_20_30"].to_numpy(),
        signalma,
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.momentum
def test_kst_readonly_input(prices_random_walk) -> None:
    arr = prices_random_walk.copy()
    arr.setflags(write=False)
    expected = kst_numpy(prices_random_walk)
    for res, exp in zip(kst_numpy(arr), expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
