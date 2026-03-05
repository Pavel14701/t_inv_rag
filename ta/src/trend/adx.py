# -*- coding: utf-8 -*-
from typing import cast

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..ma import ma_mode
from ..utils import _apply_offset_fillna
from ..volatility import atr_ind


# ----------------------------------------------------------------------
# Helper for TradingView mode (tvmode=True)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _tv_dmp_dmn_adx(
    pos: np.ndarray,
    neg: np.ndarray,
    length: int,
    signal_length: int,
    scalar: float,
    atr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DMP, DMN and ADX according to TradingView logic.
    Returns (dmp, dmn, adx).
    """
    n = len(pos)
    k = scalar / atr

    # Prepare arrays
    dmp = np.empty(n, dtype=np.float64)
    dmn = np.empty(n, dtype=np.float64)
    dx = np.empty(n, dtype=np.float64)
    adx = np.empty(n, dtype=np.float64)

    # First length values: set to NaN initially
    dmp[:length - 1] = np.nan
    dmn[:length - 1] = np.nan
    dx[:length - 1] = np.nan
    adx[:length - 1] = np.nan

    # For TradingView, the first value of the smoothed series 
    # is the average of the first `length` values.
    # We'll compute cumulative sums for pos and neg, then apply 
    # RMA-like smoothing with alpha = 1/length.
    # But to exactly mimic TradingView, we need to set the initial smoothed value 
    # to the sum of first `length` raw values.
    # Then apply EWMA with alpha = 1/length.

    # Compute sum of first `length` raw pos and neg
    sum_pos = 0.0
    sum_neg = 0.0
    for i in range(length):
        sum_pos += pos[i]
        sum_neg += neg[i]

    # Smoothed DMP and DMN at index length-1
    dmp[length - 1] = k[length - 1] * sum_pos
    dmn[length - 1] = k[length - 1] * sum_neg

    # EWMA for the rest
    alpha = 1.0 / length
    for i in range(length, n):
        dmp[i] = alpha * k[i] * pos[i] + (1.0 - alpha) * dmp[i - 1]
        dmn[i] = alpha * k[i] * neg[i] + (1.0 - alpha) * dmn[i - 1]

    # Compute DX
    for i in range(length - 1, n):
        denom = dmp[i] + dmn[i]
        if denom != 0.0:
            dx[i] = scalar * abs(dmp[i] - dmn[i]) / denom
        else:
            dx[i] = np.nan

    # DX is shifted backward by `length` (TradingView style). 
    # We'll create a shifted copy.
    dx_shifted = np.full(n, np.nan, dtype=np.float64)
    if n > length:
        dx_shifted[:-length] = dx[length:]

    # ADX is RMA of the shifted DX with period signal_length
    # First ADX value at index signal_length-1 (after shift)
    # We'll use a separate loop for RMA of DX_shifted.
    # First, compute SMA of first signal_length shifted DX values.
    adx_start_idx = length + signal_length - 1
    if adx_start_idx < n:
        # Initial SMA
        sum_dx = 0.0
        for i in range(length, length + signal_length):
            sum_dx += dx_shifted[i]
        adx[adx_start_idx] = sum_dx / signal_length

        # RMA recurrence
        alpha_sig = 1.0 / signal_length
        for i in range(adx_start_idx + 1, n):
            adx[i] = alpha_sig * dx_shifted[i] + (1.0 - alpha_sig) * adx[i - 1]

    return dmp, dmn, adx


# ----------------------------------------------------------------------
# Core ADX calculation
# ----------------------------------------------------------------------
def adx_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    signal_length: int | None = None,
    adxr_length: int = 2,
    scalar: float = 100.0,
    tvmode: bool = False,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy‑based ADX calculation.

    Returns (adx, adxr, dmp, dmn) as numpy arrays.
    """
    if signal_length is None:
        signal_length = length

    # Ensure contiguous
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # 1. ATR
    atr = atr_ind(
        high, low, close, 
        length=length, 
        mamode="rma", 
        drift=drift, 
        offset=0, 
        fillna=None, 
        use_talib=use_talib
    )
    if np.all(np.isnan(atr)):
        raise ValueError("ATR calculation failed")
    # 2. Up and Down movements
    up = high - np.roll(high, drift)
    up[:drift] = np.nan
    dn = np.roll(low, drift) - low
    dn[:drift] = np.nan
    # 3. Positive and negative directional movement
    pos = np.where((up > dn) & (up > 0), up, 0.0).astype(np.float64)
    neg = np.where((dn > up) & (dn > 0), dn, 0.0).astype(np.float64)
    # 4. Compute DMP, DMN, ADX
    if use_talib and talib_available and not tvmode:
        # TA‑Lib does not support tvmode, so fallback 
        # to our implementation if tvmode=True
        dmp = talib.PLUS_DM(high, low, timeperiod=length)
        dmn = talib.MINUS_DM(high, low, timeperiod=length)
        adx = talib.ADX(high, low, close, timeperiod=length)
    else:
        if tvmode:
            dmp, dmn, adx = _tv_dmp_dmn_adx(
                pos, neg, length, signal_length, scalar, atr
            )
        else:
            # Standard calculation using MA of pos/neg (with mamode)
            k = scalar / atr
            dmp = k * cast(np.ndarray, ma_mode(
                mamode, pos, length=length, offset=0, fillna=None, use_talib=False
            ))
            dmn = k * cast(np.ndarray, ma_mode(
                mamode, neg, length=length, offset=0, fillna=None, use_talib=False
            ))
            # DX = 100 * |DMP - DMN| / (DMP + DMN)
            denom = dmp + dmn
            with np.errstate(divide='ignore', invalid='ignore'):
                dx = scalar * np.abs(dmp - dmn) / denom
                dx = np.where(denom == 0.0, np.nan, dx)
            # ADX = MA of DX with period signal_length
            adx = cast(np.ndarray, ma_mode(
                mamode, dx, length=signal_length, offset=0, fillna=None, use_talib=False
            ))
    # 5. ADXR
    adx_shifted = np.roll(adx, adxr_length)
    adx_shifted[:adxr_length] = np.nan
    adxr = 0.5 * (adx + adx_shifted)
    # 6. Apply offset and fillna
    adx = _apply_offset_fillna(adx, offset, fillna)
    adxr = _apply_offset_fillna(adxr, offset, fillna)
    dmp = _apply_offset_fillna(dmp, offset, fillna)
    dmn = _apply_offset_fillna(dmn, offset, fillna)
    return adx, adxr, dmp, dmn


def adx_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    signal_length: int | None = None,
    adxr_length: int = 2,
    scalar: float = 100.0,
    tvmode: bool = False,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal ADX (accepts numpy arrays or Polars Series).
    Returns (adx, adxr, dmp, dmn) as numpy arrays.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return adx_numpy(
        high, low, close,
        length=length,
        signal_length=signal_length,
        adxr_length=adxr_length,
        scalar=scalar,
        tvmode=tvmode,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )


def adx_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    length: int = 14,
    signal_length: int | None = None,
    adxr_length: int = 2,
    scalar: float = 100.0,
    tvmode: bool = False,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = "",
) -> pl.DataFrame:
    """
    ADX DataFrame.

    Columns added:
        ADX_{signal_length}
        ADXR_{signal_length}_{adxr_length}
        DMP_{length}
        DMN_{length}

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col, close_col : str
        Column names for prices.
    length, signal_length, adxr_length, scalar, tvmode, \
        mamode, drift, offset, fillna, use_talib : as above.
    suffix : str
        Custom suffix (default f"_{signal_length}" etc., but internal names are fixed).

    Returns
    -------
    pl.DataFrame
    """
    if signal_length is None:
        signal_length = length

    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()

    adx_arr, adxr_arr, dmp_arr, dmn_arr = adx_numpy(
        high, low, close,
        length=length,
        signal_length=signal_length,
        adxr_length=adxr_length,
        scalar=scalar,
        tvmode=tvmode,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )
    suffix = suffix or f"_{signal_length}"
    return pl.DataFrame({
        date_col: df[date_col],
        f"ADX{suffix}": adx_arr,
        f"ADXR_{signal_length}_{adxr_length}": adxr_arr,
        f"DMP_{length}": dmp_arr,
        f"DMN_{length}": dmn_arr,
    })