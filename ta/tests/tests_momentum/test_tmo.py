# -*- coding: utf-8 -*-
"""Unit tests for True Momentum Oscillator (TMO)."""

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose, assert_array_equal

from ta.src.momentum.tmo import _tmo_main_numba, tmo_ind, tmo_numpy, tmo_polars
from ta.src.overlap.ema import ema_ind


@pytest.fixture
def oc(prices_random_walk: np.ndarray):
    np.random.seed(42)
    open_ = prices_random_walk - np.random.rand(200) * 0.5
    return open_, prices_random_walk


def _tmo_reference(
    open_: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    drift: int = 1,
    smooth: int = 4,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure numpy TMO reference."""
    n = len(close)
    mom = np.full(n, np.nan)
    mom[drift:] = open_[drift:] - close[:-drift]
    main = np.full(n, np.nan)
    for i in range(length + drift - 1, n):
        window = mom[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        main[i] = window.sum()
    # Signal with the same first-valid seeding as the implementation.
    filled = main.copy()
    first_valid = np.argmax(~np.isnan(main))
    if not np.isnan(main[first_valid]):
        filled[:first_valid] = main[first_valid]
    signalma = ema_ind(
        filled, length=smooth, use_talib=False, nan_policy="ignore"
    )
    signalma[: first_valid + smooth - 1] = np.nan
    if normalize:
        main = main * 100.0 / length
        signalma = signalma * 100.0 / length
    return main, signalma


@pytest.mark.momentum
def test_tmo_matches_reference(oc) -> None:
    open_, close = oc
    expected_main, expected_signal = _tmo_reference(open_, close)
    main, signalma = tmo_numpy(open_, close)
    assert_allclose(main, expected_main, rtol=1e-12, equal_nan=True)
    assert_allclose(signalma, expected_signal, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_tmo_warmup_nan(oc) -> None:
    open_, close = oc
    main, signalma = tmo_numpy(open_, close)
    assert np.isnan(main[:14]).all()
    assert not np.isnan(main[14:]).any()
    assert np.isnan(signalma[:17]).all()
    assert not np.isnan(signalma[17:]).any()


@pytest.mark.momentum
def test_tmo_main_is_rolling_sum(oc) -> None:
    open_, close = oc
    main, _ = tmo_numpy(open_, close, normalize=False)
    mom = np.full(200, np.nan)
    mom[1:] = open_[1:] - close[:-1]
    expected = np.full(200, np.nan)
    for i in range(14, 200):
        expected[i] = mom[i - 13 : i + 1].sum()
    assert_allclose(main, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_tmo_kernel_bitwise_vs_numpy(oc) -> None:
    open_, close = oc
    open_ = np.ascontiguousarray(open_)
    assert_array_equal(
        _tmo_main_numba(open_, close, 14, 1),
        tmo_numpy(open_, close, normalize=False)[0],
    )


@pytest.mark.momentum
def test_tmo_normalize_scales(oc) -> None:
    open_, close = oc
    raw_main, raw_signal = tmo_numpy(open_, close, normalize=False)
    norm_main, norm_signal = tmo_numpy(open_, close, normalize=True)
    scale = 100.0 / 14
    assert_allclose(norm_main, raw_main * scale, rtol=1e-12, equal_nan=True)
    assert_allclose(
        norm_signal, raw_signal * scale, rtol=1e-12, equal_nan=True
    )


@pytest.mark.momentum
def test_tmo_rising_open_negative_main() -> None:
    # open >> close every bar (gaps up), close strictly falling with
    # step ~ -5.76: mom = open[i] - close[i-1] = 5 - 5.76 < 0 -> bearish.
    close = np.linspace(350.0, 10.0, 60)
    open_ = close + 5.0
    main, _ = tmo_numpy(open_, close, normalize=False)
    assert (main[14:] < 0.0).all()


@pytest.mark.momentum
def test_tmo_falling_open_positive_main() -> None:
    # open << close every bar (gaps down), close strictly rising with
    # step ~ +5.76: mom = open[i] - close[i-1] = -5 + 5.76 > 0 -> bullish.
    close = np.linspace(10.0, 350.0, 60)
    open_ = close - 5.0
    main, _ = tmo_numpy(open_, close, normalize=False)
    assert (main[14:] > 0.0).all()


@pytest.mark.momentum
def test_tmo_nan_propagation(oc) -> None:
    open_, close = oc
    close = close.copy()
    close[50] = np.nan
    main, _ = tmo_numpy(open_, close)
    # mom[j] = open[j] - close[j-1] is NaN for j = 51; windows
    # [i-13, i] include bar 51 for i in [51, 64].
    assert np.isnan(main[51:65]).all()
    assert not np.isnan(main[14:51]).any()
    assert not np.isnan(main[65:]).any()


@pytest.mark.momentum
@pytest.mark.parametrize(
    "kw",
    [
        {"length": 0},
        {"drift": 0},
        {"smooth": -1},
    ],
)
def test_tmo_invalid_params(kw: dict) -> None:
    with pytest.raises(ValueError):
        tmo_numpy(np.arange(40.0), np.arange(40.0), **kw)


@pytest.mark.momentum
def test_tmo_empty_input() -> None:
    empty = np.array([], dtype=np.float64)
    main, signalma = tmo_numpy(empty, empty)
    assert main.size == 0
    assert signalma.size == 0


@pytest.mark.momentum
def test_tmo_offset_fillna(oc) -> None:
    open_, close = oc
    base_main, base_signal = tmo_numpy(open_, close)
    main, signalma = tmo_numpy(open_, close, offset=2, fillna=0.0)
    # fillna replaces both shifted-in positions and warm-up NaNs.
    exp_main = np.where(np.isnan(base_main), 0.0, base_main)
    exp_signal = np.where(np.isnan(base_signal), 0.0, base_signal)
    assert_array_equal(main[:2], np.zeros(2))
    assert_allclose(main[2:], exp_main[:-2], rtol=1e-12)
    assert_allclose(signalma[2:], exp_signal[:-2], rtol=1e-12)


@pytest.mark.momentum
def test_tmo_ind_numpy_and_series(oc) -> None:
    open_, close = oc
    expected = tmo_numpy(open_, close)
    from_arrays = tmo_ind(open_, close)
    from_series = tmo_ind(pl.Series(open_), pl.Series(close))
    for res, exp in zip(from_arrays, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)
    for res, exp in zip(from_series, expected):
        assert_allclose(res, exp, rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_tmo_polars(df_ohlc: pl.DataFrame) -> None:
    # df_ohlc has no 'open' column -> synthesise one from high/low.
    df = df_ohlc.with_columns(
        ((df_ohlc["high"] + df_ohlc["low"]) / 2.0).alias("open")
    )
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    main, signalma = tmo_numpy(open_, close)
    result = tmo_polars(df)
    assert "TMO_14" in result.columns
    assert "TMOS_14" in result.columns
    assert_allclose(
        result["TMO_14"].to_numpy(), main, rtol=1e-12, equal_nan=True
    )
    assert_allclose(
        result["TMOS_14"].to_numpy(), signalma, rtol=1e-12, equal_nan=True
    )


@pytest.mark.momentum
def test_tmo_readonly_input(oc) -> None:
    open_, close = oc
    expected = tmo_numpy(open_, close)
    open_ = open_.copy()
    open_.setflags(write=False)
    main, signalma = tmo_numpy(open_, close)
    assert_allclose(main, expected[0], rtol=1e-12, equal_nan=True)
    assert_allclose(signalma, expected[1], rtol=1e-12, equal_nan=True)


@pytest.mark.momentum
def test_tmo_drift(oc) -> None:
    open_, close = oc
    main, _ = tmo_numpy(open_, close, drift=3, normalize=False)
    mom = np.full(200, np.nan)
    mom[3:] = open_[3:] - close[:-3]
    expected = np.full(200, np.nan)
    for i in range(16, 200):
        expected[i] = mom[i - 13 : i + 1].sum()
    assert_allclose(main, expected, rtol=1e-12, equal_nan=True)
