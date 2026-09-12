# -*- coding: utf-8 -*-
"""Unit tests for Ichimoku Kinko Hyo (Ichimoku Cloud) indicator.

Tests cover:
- _midprice_multi_numba against a pure-Python reference
- NaN propagation inside windows (IEEE 754)
- _shift_forward behaviour
- ichimoku_core_numba against reference (incl. Chikou Span)
- ichimoku_ind: columns, Senkou displacement, Chikou, forward DataFrame
  (temporal and integer index), offset/fillna, validation, nan_policy
- IEEE 754 compliance (NaN, Inf, empty, extreme)
"""

from datetime import date, timedelta

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from numpy.testing import assert_allclose

from ta.src._array_ops import _apply_offset_fillna
from ta.src.overlap.ichimoku import (
    _midprice_multi_numba,
    _shift_forward,
    ichimoku_core_numba,
    ichimoku_ind,
)


# -----------------------------------------------------------------------------
# Helpers to build synthetic OHLC data
# -----------------------------------------------------------------------------
def _make_ohlc(
    close: npt.NDArray[np.float64],
    seed: int = 42,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build high/low arrays around a close series."""
    rng = np.random.default_rng(seed)
    spread = np.abs(rng.standard_normal(len(close))) * 0.5 + 0.1
    high = close + spread
    low = close - spread
    return high, low


def _make_ohlc_df(
    close: npt.NDArray[np.float64],
    with_dates: bool = False,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a Polars DataFrame with high/low/close (and optional dates)."""
    high, low = _make_ohlc(close, seed)
    data = {"high": high, "low": low, "close": close}
    if with_dates:
        data["date"] = pl.date_range(
            date(2020, 1, 1),
            date(2020, 1, 1) + timedelta(days=len(close) - 1),
            "1d",
            eager=True,
        )
    return pl.DataFrame(data)


# -----------------------------------------------------------------------------
# Reference implementation (pure Python)
# -----------------------------------------------------------------------------
def _midprice_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    length: int,
) -> npt.NDArray[np.float64]:
    """Pure Python rolling midprice with NaN propagation."""
    n = len(high)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(length - 1, n):
        wh = high[i - length + 1 : i + 1]
        wl = low[i - length + 1 : i + 1]
        if np.isnan(wh).any() or np.isnan(wl).any():
            continue
        out[i] = (np.max(wh) + np.min(wl)) * 0.5
    return out


def _ichimoku_reference(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    tenkan: int,
    kijun: int,
    senkou: int,
    include_chikou: bool = True,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Pure Python reference Ichimoku (unshifted spans + chikou)."""
    tenkan_sen = _midprice_reference(high, low, tenkan)
    kijun_sen = _midprice_reference(high, low, kijun)
    span_a = (tenkan_sen + kijun_sen) * 0.5
    span_b = _midprice_reference(high, low, senkou)
    n = len(close)
    chikou = np.full(n, np.nan, dtype=np.float64)
    if include_chikou and kijun < n:
        chikou[:-kijun] = np.asarray(close, dtype=np.float64)[kijun:]
    return tenkan_sen, kijun_sen, span_a, span_b, chikou


# -----------------------------------------------------------------------------
# Tests for _midprice_multi_numba
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_midprice_multi_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Midprices for three lengths must match the pure-Python reference."""
    close = prices_random_walk
    high, low = _make_ohlc(close)
    out1, out2, out3 = _midprice_multi_numba(high, low, 9, 26, 52)
    assert_allclose(
        out1, _midprice_reference(high, low, 9), rtol=1e-12, equal_nan=True
    )
    assert_allclose(
        out2, _midprice_reference(high, low, 26), rtol=1e-12, equal_nan=True
    )
    assert_allclose(
        out3, _midprice_reference(high, low, 52), rtol=1e-12, equal_nan=True
    )


@pytest.mark.overlap
def test_midprice_multi_warmup_nan() -> None:
    """First (length - 1) values of each output must be NaN."""
    close = np.linspace(1.0, 20.0, 20)
    high, low = _make_ohlc(close)
    out1, out2, out3 = _midprice_multi_numba(high, low, 3, 5, 10)
    assert np.isnan(out1[:2]).all() and np.isfinite(out1[2:]).all()
    assert np.isnan(out2[:4]).all() and np.isfinite(out2[4:]).all()
    assert np.isnan(out3[:9]).all() and np.isfinite(out3[9:]).all()


@pytest.mark.overlap
def test_midprice_multi_short_input() -> None:
    """Input shorter than the longest window returns all NaN."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    high, low = _make_ohlc(close)
    out1, out2, out3 = _midprice_multi_numba(high, low, 3, 5, 10)
    for out in (out1, out2, out3):
        assert np.isnan(out).all()
        assert len(out) == len(close)


@pytest.mark.overlap
def test_midprice_multi_nan_propagation() -> None:
    """NaN anywhere inside a window must make the result NaN (IEEE 754)."""
    close = np.linspace(1.0, 15.0, 15)
    high, low = _make_ohlc(close)
    # NaN in the middle of the first window (index 1, window 0..2)
    high_bad = high.copy()
    high_bad[1] = np.nan
    out1, _, _ = _midprice_multi_numba(high_bad, low, 3, 5, 10)
    # windows of length 3 covering index 1: results at 1..3 are NaN
    assert np.isnan(out1[1:4]).all()
    assert np.isfinite(out1[4:]).all()

    # NaN at the start of a window
    low_bad = low.copy()
    low_bad[5] = np.nan
    _, out2, _ = _midprice_multi_numba(high, low_bad, 3, 5, 10)
    # windows of length 5 covering index 5: results at 5..9 are NaN
    assert np.isnan(out2[5:10]).all()
    assert np.isfinite(out2[10:]).all()


# -----------------------------------------------------------------------------
# Tests for _shift_forward
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_shift_forward_basic() -> None:
    """Positive shift moves values forward and fills the head with NaN."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = _shift_forward(arr, 2)
    assert np.isnan(out[:2]).all()
    assert_allclose(out[2:], arr[:-2])


@pytest.mark.overlap
def test_shift_forward_zero_and_negative() -> None:
    """Zero or negative shift returns a copy of the input."""
    arr = np.array([1.0, 2.0, 3.0])
    out0 = _shift_forward(arr, 0)
    outneg = _shift_forward(arr, -3)
    assert_allclose(out0, arr)
    assert_allclose(outneg, arr)
    # must be copies, not views
    out0[0] = -1.0
    assert arr[0] == 1.0


@pytest.mark.overlap
def test_shift_forward_shift_ge_n() -> None:
    """Shift >= len(arr) returns all NaN."""
    arr = np.array([1.0, 2.0, 3.0])
    out = _shift_forward(arr, 3)
    assert np.isnan(out).all()
    out2 = _shift_forward(arr, 10)
    assert np.isnan(out2).all()


# -----------------------------------------------------------------------------
# Tests for ichimoku_core_numba
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_core_against_reference(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Core outputs must match the pure-Python reference."""
    close = prices_random_walk
    high, low = _make_ohlc(close)
    tenkan, kijun, senkou = 9, 26, 52
    t_nb, k_nb, a_nb, b_nb, c_nb = ichimoku_core_numba(
        high, low, close, tenkan, kijun, senkou
    )
    t_ref, k_ref, a_ref, b_ref, c_ref = _ichimoku_reference(
        high, low, close, tenkan, kijun, senkou
    )
    assert_allclose(t_nb, t_ref, rtol=1e-12, equal_nan=True)
    assert_allclose(k_nb, k_ref, rtol=1e-12, equal_nan=True)
    assert_allclose(a_nb, a_ref, rtol=1e-12, equal_nan=True)
    assert_allclose(b_nb, b_ref, rtol=1e-12, equal_nan=True)
    assert_allclose(c_nb, c_ref, rtol=1e-12, equal_nan=True)


@pytest.mark.overlap
def test_core_chikou_is_close_shifted_back() -> None:
    """Chikou[i] must equal close[i + kijun]; last kijun values are NaN."""
    close = np.linspace(10.0, 60.0, 50)
    high, low = _make_ohlc(close)
    kijun = 5
    _, _, _, _, chikou = ichimoku_core_numba(high, low, close, 2, kijun, 8)
    assert chikou is not None
    assert_allclose(chikou[:-kijun], close[kijun:])
    assert np.isnan(chikou[-kijun:]).all()


@pytest.mark.overlap
def test_core_no_chikou() -> None:
    """include_chikou=False or lookahead=False -> chikou is None."""
    close = np.linspace(1.0, 30.0, 30)
    high, low = _make_ohlc(close)
    res1 = ichimoku_core_numba(high, low, close, 2, 3, 5, include_chikou=False)
    res2 = ichimoku_core_numba(
        high, low, close, 2, 3, 5, include_chikou=True, lookahead=False
    )
    assert res1[4] is None
    assert res2[4] is None


@pytest.mark.overlap
def test_core_short_input_graceful() -> None:
    """Core handles short input gracefully (all NaN, no exception)."""
    close = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    high, low = _make_ohlc(close)
    t, k, a, b, c = ichimoku_core_numba(high, low, close, 5, 10, 20)
    for out in (t, k, a, b):
        assert np.isnan(out).all()
    assert c is not None and np.isnan(c).all()


@pytest.mark.overlap
def test_core_invalid_periods() -> None:
    """Periods < 1 raise ValueError."""
    close = np.linspace(1.0, 30.0, 30)
    high, low = _make_ohlc(close)
    with pytest.raises(ValueError, match="must all be >= 1"):
        ichimoku_core_numba(high, low, close, 0, 3, 5)
    with pytest.raises(ValueError, match="must all be >= 1"):
        ichimoku_core_numba(high, low, close, 3, -1, 5)


@pytest.mark.overlap
def test_core_length_mismatch() -> None:
    """Mismatched array lengths raise ValueError."""
    close = np.linspace(1.0, 30.0, 30)
    high, low = _make_ohlc(close)
    with pytest.raises(ValueError, match="same length"):
        ichimoku_core_numba(high, low, close[:-1], 2, 3, 5)


# -----------------------------------------------------------------------------
# Tests for ichimoku_ind (Polars integration)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_ind_columns_and_values(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Historical columns match the reference with Senkou displacement."""
    close = prices_random_walk
    df = _make_ohlc_df(close)
    tenkan, kijun, senkou = 9, 26, 52
    hist, fwd = ichimoku_ind(
        df, date_col=None, tenkan=tenkan, kijun=kijun, senkou=senkou
    )
    expected_cols = [
        f"ITS_{tenkan}",
        f"IKS_{kijun}",
        f"ISA_{tenkan}",
        f"ISB_{senkou}",
        f"ICS_{kijun}",
    ]
    for col in expected_cols:
        assert col in hist.columns
        assert hist[col].dtype == pl.Float64
    assert len(hist) == len(df)

    t_ref, k_ref, a_ref, b_ref, c_ref = _ichimoku_reference(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        tenkan,
        kijun,
        senkou,
    )
    # Tenkan / Kijun are not shifted
    assert_allclose(
        hist[f"ITS_{tenkan}"].to_numpy(), t_ref, rtol=1e-12, equal_nan=True
    )
    assert_allclose(
        hist[f"IKS_{kijun}"].to_numpy(), k_ref, rtol=1e-12, equal_nan=True
    )
    # Senkou spans are shifted forward by kijun
    assert_allclose(
        hist[f"ISA_{tenkan}"].to_numpy()[kijun:],
        a_ref[:-kijun],
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        hist[f"ISB_{senkou}"].to_numpy()[kijun:],
        b_ref[:-kijun],
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.isnan(hist[f"ISA_{tenkan}"].to_numpy()[:kijun]).all()
    # Chikou is close shifted backward by kijun
    assert_allclose(
        hist[f"ICS_{kijun}"].to_numpy()[:-kijun],
        close[kijun:],
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.isnan(hist[f"ICS_{kijun}"].to_numpy()[-kijun:]).all()

    # Forward DataFrame: last kijun unshifted span values
    assert len(fwd) == kijun
    assert_allclose(
        fwd[f"ISA_{tenkan}"].to_numpy(),
        a_ref[-kijun:],
        rtol=1e-12,
        equal_nan=True,
    )
    assert_allclose(
        fwd[f"ISB_{senkou}"].to_numpy(),
        b_ref[-kijun:],
        rtol=1e-12,
        equal_nan=True,
    )


@pytest.mark.overlap
def test_ind_forward_with_dates(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Temporal date column produces future dates in the forward DataFrame."""
    close = prices_random_walk
    df = _make_ohlc_df(close, with_dates=True)
    kijun = 5
    _, fwd = ichimoku_ind(df, tenkan=3, kijun=kijun, senkou=8, date_col="date")
    assert "date" in fwd.columns
    assert fwd["date"].dtype == pl.Date
    assert len(fwd) == kijun
    last_date = df["date"].item(-1)
    assert fwd["date"].item(0) == last_date + timedelta(days=1)
    assert fwd["date"].item(-1) == last_date + timedelta(days=kijun)


@pytest.mark.overlap
def test_ind_forward_non_temporal_date_col(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Non-temporal date column falls back to an integer index (bug fix)."""
    close = prices_random_walk
    df = _make_ohlc_df(close).with_row_index("date")
    kijun = 5
    _, fwd = ichimoku_ind(df, tenkan=3, kijun=kijun, senkou=8, date_col="date")
    # must not raise; integer index starts after the last row
    assert "index" in fwd.columns
    assert len(fwd) == kijun
    assert fwd["index"].item(0) == len(df)
    assert fwd["index"].item(-1) == len(df) + kijun - 1


@pytest.mark.overlap
def test_ind_forward_integer_index_no_date_col(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """A None date_col produces an integer index forward DataFrame."""
    close = prices_random_walk
    df = _make_ohlc_df(close)
    kijun = 5
    _, fwd = ichimoku_ind(df, date_col=None, tenkan=3, kijun=kijun, senkou=8)
    assert "index" in fwd.columns
    assert fwd["index"].item(0) == len(df)


@pytest.mark.overlap
def test_ind_offset_fillna(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Offset and fillna are applied to all historical components."""
    close = prices_random_walk
    df = _make_ohlc_df(close)
    tenkan, kijun, senkou = 3, 5, 8
    offset = 2
    fillna = 0.0
    hist_base, _ = ichimoku_ind(
        df, date_col=None, tenkan=tenkan, kijun=kijun, senkou=senkou
    )
    hist_off, _ = ichimoku_ind(
        df,
        date_col=None,
        tenkan=tenkan,
        kijun=kijun,
        senkou=senkou,
        offset=offset,
        fillna=fillna,
    )
    for col in hist_base.columns:
        expected = _apply_offset_fillna(
            np.array(hist_base[col].to_numpy()), offset, fillna
        )
        assert_allclose(
            hist_off[col].to_numpy(), expected, rtol=1e-12, equal_nan=True
        )
        # fillna replaced all NaNs
        assert np.isfinite(hist_off[col].to_numpy()).all()


@pytest.mark.overlap
def test_ind_no_chikou_column(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """include_chikou=False / lookahead=False omit the ICS column."""
    close = prices_random_walk
    df = _make_ohlc_df(close)
    hist1, _ = ichimoku_ind(
        df,
        date_col=None,
        tenkan=3,
        kijun=5,
        senkou=8,
        include_chikou=False,
    )
    hist2, _ = ichimoku_ind(
        df, date_col=None, tenkan=3, kijun=5, senkou=8, lookahead=False
    )
    assert "ICS_5" not in hist1.columns
    assert "ICS_5" not in hist2.columns


@pytest.mark.overlap
def test_ind_custom_column_names(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Custom high/low/close column names are respected."""
    close = prices_random_walk
    df = _make_ohlc_df(close).rename(
        {
            "high": "H",
            "low": "L",
            "close": "C",
        }
    )
    hist, _ = ichimoku_ind(
        df,
        high_col="H",
        low_col="L",
        close_col="C",
        date_col=None,
        tenkan=3,
        kijun=5,
        senkou=8,
    )
    assert "ITS_3" in hist.columns
    assert len(hist) == len(df)


@pytest.mark.overlap
def test_ind_validation_errors(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Invalid periods, mismatched columns and short data raise ValueError."""
    close = prices_random_walk
    df = _make_ohlc_df(close)
    with pytest.raises(ValueError, match="must all be >= 1"):
        ichimoku_ind(df, tenkan=0, kijun=5, senkou=8)
    with pytest.raises(ValueError, match="too short"):
        ichimoku_ind(df.head(5), tenkan=3, kijun=5, senkou=8)


@pytest.mark.overlap
def test_ind_nan_policy_raise(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Input with NaN and nan_policy='raise' raises ValueError."""
    close = prices_random_walk.copy()
    close[10] = np.nan
    df = _make_ohlc_df(close)
    with pytest.raises(ValueError, match="NaN"):
        ichimoku_ind(df, tenkan=3, kijun=5, senkou=8)


@pytest.mark.overlap
def test_ind_nan_policy_ffill(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """Input with NaN and nan_policy='ffill' is filled and computed."""
    close = prices_random_walk.copy()
    close[10] = np.nan
    df = _make_ohlc_df(close)
    hist, _ = ichimoku_ind(
        df, date_col=None, tenkan=3, kijun=5, senkou=8, nan_policy="ffill"
    )
    assert len(hist) == len(df)
    # after warmup all values must be finite
    assert np.isfinite(hist["ITS_3"].to_numpy()[2:]).all()


@pytest.mark.overlap
def test_ind_nan_policy_ignore_propagates(
    prices_random_walk: npt.NDArray[np.float64],
) -> None:
    """NaN input with nan_policy='ignore' propagates NaN into windows."""
    close = prices_random_walk.copy()
    close[10] = np.nan
    df = _make_ohlc_df(close)
    hist, _ = ichimoku_ind(
        df, date_col=None, tenkan=3, kijun=5, senkou=8, nan_policy="ignore"
    )
    its = hist["ITS_3"].to_numpy()
    # windows covering index 10 (length 3): results at 10, 11, 12 are NaN
    assert np.isnan(its[10:13]).all()
    assert np.isfinite(its[13:]).all()


# -----------------------------------------------------------------------------
# IEEE 754 compliance tests (using fixtures from conftest.py)
# -----------------------------------------------------------------------------
@pytest.mark.overlap
def test_ind_with_inf(prices_with_inf: npt.NDArray[np.float64]) -> None:
    """Input Inf is replaced with NaN and handled per nan_policy."""
    df = _make_ohlc_df(prices_with_inf)
    # 'raise' -> Inf becomes NaN -> raises
    with pytest.raises(ValueError, match="NaN"):
        ichimoku_ind(df, tenkan=2, kijun=3, senkou=4)
    # 'ffill' -> works fine
    hist, _ = ichimoku_ind(
        df, date_col=None, tenkan=2, kijun=3, senkou=4, nan_policy="ffill"
    )
    assert len(hist) == len(df)
    assert np.isfinite(hist["ITS_2"].to_numpy()[1:]).all()


@pytest.mark.overlap
def test_ind_empty(prices_empty: npt.NDArray[np.float64]) -> None:
    """Empty input raises (series too short)."""
    df = _make_ohlc_df(prices_empty)
    with pytest.raises(ValueError, match="too short"):
        ichimoku_ind(df, tenkan=2, kijun=3, senkou=4)


@pytest.mark.overlap
def test_ind_all_nan(prices_all_nan: npt.NDArray[np.float64]) -> None:
    """All-NaN input: 'raise' raises, fillna fills everything."""
    df = _make_ohlc_df(prices_all_nan)
    with pytest.raises(ValueError, match="NaN"):
        ichimoku_ind(df, tenkan=2, kijun=3, senkou=4)
    hist, _ = ichimoku_ind(
        df,
        date_col=None,
        tenkan=2,
        kijun=3,
        senkou=4,
        nan_policy="ignore",
        fillna=0.0,
    )
    for col in hist.columns:
        assert (hist[col].to_numpy() == 0.0).all()


@pytest.mark.overlap
def test_ind_extreme_values(
    prices_extreme: npt.NDArray[np.float64],
) -> None:
    """Extreme values (1e300, 1e-300) must not crash."""
    df = _make_ohlc_df(prices_extreme)
    hist, _ = ichimoku_ind(
        df,
        date_col=None,
        tenkan=2,
        kijun=3,
        senkou=4,
        nan_policy="ignore",
    )
    assert hist is not None
    assert len(hist) == len(df)


@pytest.mark.overlap
def test_ind_with_nan_fixture(
    prices_with_nan: npt.NDArray[np.float64],
) -> None:
    """The NaN fixture propagates through the indicator without crashing."""
    df = _make_ohlc_df(prices_with_nan)
    hist, _ = ichimoku_ind(
        df, date_col=None, tenkan=2, kijun=3, senkou=4, nan_policy="ignore"
    )
    its = hist["ITS_2"].to_numpy()
    # NaN at index 5 affects windows ending at 5 and 6
    assert np.isnan(its[5:7]).all()
    assert np.isfinite(its[7:]).all()
