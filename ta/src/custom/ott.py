# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import polars as pl
from numba import float64, int64, njit

from ..overlap import ema_ind, sma_ind, wma_ind
from ..utils import _apply_offset_fillna, _handle_nan_policy


# ----------------------------------------------------------------------
# Специфические MA функции (Numba)
# ----------------------------------------------------------------------
@njit(float64[:](float64[:], int64), fastmath=True, cache=True)
def _wwma_numba(price: np.ndarray, length: int) -> np.ndarray:
    """Welles Wilder Moving Average (simple exponential smoothing)."""
    n = len(price)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    alpha = 1.0 / length
    out[0] = price[0]
    for i in range(1, n):
        out[i] = alpha * price[i] + (1 - alpha) * out[i - 1]
    return out


@njit(float64[:](float64[:], int64), fastmath=True, cache=True)
def _vidya_numba(price: np.ndarray, length: int) -> np.ndarray:
    """Variable Index Dynamic Average (VIDYA) with CMO period 9."""
    n = len(price)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 10:
        return out
    alpha = 2.0 / (length + 1.0)

    # CMO with period 9
    up_sum = np.zeros(n, dtype=np.float64)
    down_sum = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = price[i] - price[i - 1]
        if diff > 0:
            up_sum[i] = diff
        elif diff < 0:
            down_sum[i] = -diff

    cmo = np.full(n, np.nan, dtype=np.float64)
    for i in range(8, n):
        up_roll = 0.0
        down_roll = 0.0
        for j in range(i - 8, i + 1):
            up_roll += up_sum[j]
            down_roll += down_sum[j]
        total = up_roll + down_roll
        cmo[i] = (up_roll - down_roll) / total if total != 0 else 0.0
    # VIDYA
    out[0] = price[0]
    for i in range(1, n):
        k = alpha * abs(cmo[i]) if not np.isnan(cmo[i]) else 0
        out[i] = k * price[i] + (1 - k) * out[i - 1]
    return out


@njit(float64[:](float64[:], int64), fastmath=True, cache=True)
def _tsf_numba(price: np.ndarray, length: int) -> np.ndarray:
    """Time Series Forecast (linear regression extrapolated one bar)."""
    n = len(price)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    # Precompute constants for linear regression
    sum_x = 0.0
    sum_x2 = 0.0
    for j in range(length):
        sum_x += j
        sum_x2 += j * j
    inv_len = 1.0 / length
    denom = sum_x2 - sum_x * sum_x * inv_len
    for i in range(length - 1, n):
        sum_y = 0.0
        sum_xy = 0.0
        for j in range(length):
            val = price[i - length + 1 + j]
            sum_y += val
            sum_xy += j * val
        b = (sum_xy - sum_x * sum_y * inv_len) / denom
        a = (sum_y - b * sum_x) * inv_len
        out[i] = a + b * length  # forecast next bar
    return out


def _tma_numpy(
    price: np.ndarray, 
    length: int, 
    use_talib: bool, 
    nan_policy: str
) -> np.ndarray:
    """Triangular Moving Average."""
    first_len = int(np.ceil(length / 2))
    second_len = int(np.floor(length / 2)) + 1
    sma1 = sma_ind(price, length=first_len, use_talib=use_talib, nan_policy=nan_policy)
    return sma_ind(sma1, length=second_len, use_talib=use_talib, nan_policy=nan_policy)


def _zlema_numpy(
    price: np.ndarray, 
    length: int, 
    use_talib: bool, 
    nan_policy: str
) -> np.ndarray:
    """Zero-Lag EMA."""
    lag = int(length / 2)
    shifted = np.roll(price, lag)
    shifted[:lag] = np.nan
    zlema_data = price + (price - shifted)
    return ema_ind(
        zlema_data, length=length, use_talib=use_talib, nan_policy=nan_policy
    )


# ----------------------------------------------------------------------
# Основная функция расчета OTT
# ----------------------------------------------------------------------

def ott_numpy(
    close: np.ndarray,
    length: int = 2,
    percent: float = 1.4,
    ma_type: Literal["SMA", "EMA", "WMA", "TMA", "VAR", "WWMA", "ZLEMA", "TSF"] = "VAR",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy‑based Optimized Trend Tracker (OTT).

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        Period for MA.
    percent : float
        Percentage factor for stop levels.
    ma_type : str
        Type of moving average.
    offset, fillna, use_talib, nan_policy, trim : as usual.

    Returns
    -------
    (ma, long_stop, short_stop, direction, ott)
    """
    # ---- Validation ----
    if length < 1:
        raise ValueError("length must be >= 1")
    if percent <= 0:
        raise ValueError("percent must be positive")

    close = np.asarray(close, dtype=np.float64)
    if np.isinf(close).any():
        raise ValueError("Input contains non‑finite values (inf or -inf).")
    close = _handle_nan_policy(close, nan_policy, "close")
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    n = len(close)
    if n < length:
        raise ValueError(
            f"Input series too short: need at least {length} elements, got {n}."
        )
    # ---- Moving Average ----
    ma_type_upper = ma_type.upper()
    if ma_type_upper == "SMA":
        ma = sma_ind(close, length=length, use_talib=use_talib, nan_policy=nan_policy)
    elif ma_type_upper == "EMA":
        ma = ema_ind(close, length=length, use_talib=use_talib, nan_policy=nan_policy)
    elif ma_type_upper == "WMA":
        ma = wma_ind(close, length=length, use_talib=use_talib, nan_policy=nan_policy)
    elif ma_type_upper == "TMA":
        ma = _tma_numpy(close, length, use_talib, nan_policy)
    elif ma_type_upper == "VAR":
        ma = _vidya_numba(close, length)
    elif ma_type_upper == "WWMA":
        ma = _wwma_numba(close, length)
    elif ma_type_upper == "ZLEMA":
        ma = _zlema_numpy(close, length, use_talib, nan_policy)
    elif ma_type_upper == "TSF":
        ma = _tsf_numba(close, length)
    else:
        raise ValueError(f"Unsupported ma_type: {ma_type}")
    # ---- Stop levels ----
    offset_vals = ma * percent / 100.0
    long_stop_raw = ma - offset_vals
    short_stop_raw = ma + offset_vals
    long_stop = np.maximum.accumulate(long_stop_raw)
    short_stop = np.minimum.accumulate(short_stop_raw)
    # ---- Trend direction ----
    direction = _compute_trend_direction_numba(ma, long_stop, short_stop)
    # ---- OTT ----
    base_stop = np.where(direction == 1, long_stop, short_stop)
    ott = np.where(
        ma > base_stop,
        base_stop * (200.0 + percent) / 200.0,
        base_stop * (200.0 - percent) / 200.0,
    )
    # ---- Trim ----
    if trim:
        valid = np.where(~np.isnan(ott))[0]
        if len(valid) > 0:
            start = valid[0]
            ma = ma[start:]
            long_stop = long_stop[start:]
            short_stop = short_stop[start:]
            direction = direction[start:]
            ott = ott[start:]
        else:
            ma = np.array([])
            long_stop = np.array([])
            short_stop = np.array([])
            direction = np.array([])
            ott = np.array([])
    # ---- Offset & fillna ----
    ma = _apply_offset_fillna(ma, offset, fillna)
    long_stop = _apply_offset_fillna(long_stop, offset, fillna)
    short_stop = _apply_offset_fillna(short_stop, offset, fillna)
    direction = _apply_offset_fillna(direction, offset, fillna)
    ott = _apply_offset_fillna(ott, offset, fillna)

    return ma, long_stop, short_stop, direction, ott


@njit((float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def _compute_trend_direction_numba(
    ma: np.ndarray,
    long_stop: np.ndarray,
    short_stop: np.ndarray,
) -> np.ndarray:
    """Compute trend direction (1 for up, -1 for down)."""
    n = len(ma)
    direction = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        if direction[i - 1] == -1 and ma[i] > short_stop[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and ma[i] < long_stop[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return direction


# ----------------------------------------------------------------------
# Универсальная функция (numpy / polars)
# ----------------------------------------------------------------------

def ott_ind(
    close: np.ndarray | pl.Series,
    length: int = 2,
    percent: float = 1.4,
    ma_type: Literal[
        "SMA", "EMA", "WMA", 
        "TMA", "VAR", "WWMA", 
        "ZLEMA", "TSF"
    ] = "VAR",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal OTT (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return ott_numpy(
        close,
        length=length,
        percent=percent,
        ma_type=ma_type,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=trim,
    )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def ott_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 2,
    percent: float = 1.4,
    ma_type: Literal[
        "SMA", "EMA", "WMA", 
        "TMA", "VAR", "WWMA", 
        "ZLEMA", "TSF"
    ] = "VAR",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add OTT columns to Polars DataFrame.

    Columns added:
        OTT_MA{suffix}
        OTT_LONG_STOP{suffix}
        OTT_SHORT_STOP{suffix}
        OTT_DIRECTION{suffix}
        OTT{suffix}
    """
    close = df[close_col].to_numpy()
    ma, long_stop, short_stop, direction, ott = ott_numpy(
        close,
        length=length,
        percent=percent,
        ma_type=ma_type,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,
    )
    if not suffix:
        suffix = f"_{length}_{percent}_{ma_type}"
    return df.with_columns([
        pl.Series(f"OTT_MA{suffix}", ma),
        pl.Series(f"OTT_LONG_STOP{suffix}", long_stop),
        pl.Series(f"OTT_SHORT_STOP{suffix}", short_stop),
        pl.Series(f"OTT_DIRECTION{suffix}", direction),
        pl.Series(f"OTT{suffix}", ott),
    ])