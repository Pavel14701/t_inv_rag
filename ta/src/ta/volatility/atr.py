# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..overlap import ema_ind, rma_ind, sma_ind
from ..utils import _apply_offset_fillna
from .true_range import true_range_ind


# ----------------------------------------------------------------------
# ATR – Numba implementation (TR + RMA/SMA/EMA)
# ----------------------------------------------------------------------
def atr_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False
) -> np.ndarray:
    """
    Average True Range using Numba for TR and smoothing.
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Ensure contiguous
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # True Range
    tr = true_range_ind(high, low, close, drift)
    # Smooth TR with selected MA
    mamode = mamode.lower()
    if mamode == "rma":
        atr = rma_ind(tr, length)
    elif mamode == "sma":
        atr = sma_ind(tr, length)
    elif mamode == "ema":
        atr = ema_ind(tr, length)
    else:
        raise ValueError(f"Unsupported mamode: {mamode}")
    # Convert to percent if requested
    if percent:
        atr = atr * 100.0 / close
    # Apply offset and fillna in one go
    return _apply_offset_fillna(atr, offset, fillna)


# ----------------------------------------------------------------------
# ATR – TA-Lib wrapper (when available)
# ----------------------------------------------------------------------
def atr_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False
) -> np.ndarray:
    """
    ATR using TA-Lib (C implementation).
    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")

    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    atr = talib.ATR(high, low, close, timeperiod=length)
    if percent:
        atr = atr * 100.0 / close
    return _apply_offset_fillna(atr, offset, fillna)


# ----------------------------------------------------------------------
# Universal ATR function
# ----------------------------------------------------------------------
def atr_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False,
    use_talib: bool = True
) -> np.ndarray:
    """
    Universal Average True Range with automatic backend selection.
    """
    # Convert Polars Series to numpy
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return atr_talib(
            high, low, close, length, offset, fillna, percent
        )
    else:
        return atr_numba(
            high, low, close, length, mamode, drift, offset, fillna, percent
        )


# ----------------------------------------------------------------------
# Polars integration for ATR
# ----------------------------------------------------------------------
def atr_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    length: int = 14,
    mamode: str = "rma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False,
    use_talib: bool = True,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    ATR for Polars DataFrame.
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = atr_ind(
        high, low, close,
        length=length,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        percent=percent,
        use_talib=use_talib
    )
    out_name = output_col or f"ATR_{length}"
    return df.with_columns([pl.Series(out_name, result)])