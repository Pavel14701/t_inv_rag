# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..overlap import ema_ind, rma_ind, sma_ind
from ..utils import _apply_offset_fillna, _handle_nan_policy
from .true_range import true_range_ind


# ----------------------------------------------------------------------
# ATR – Numba implementation (TR + RMA/SMA/EMA) with NaN handling and trim
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
    percent: bool = False,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """
    Average True Range using Numba for TR and smoothing.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays.
    length : int
        ATR period.
    mamode : str
        Moving average mode ('rma', 'sma', 'ema').
    drift : int
        Lookback period for true range.
    offset, fillna, percent, nan_policy, trim : as usual.

    Returns
    -------
    np.ndarray
        ATR values.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    for name, arr in [("high", high), ("low", low), ("close", close)]:
        if np.isinf(arr).any():
            raise ValueError(f"Input {name} contains non-finite values (inf or -inf).")
    high = _handle_nan_policy(high, nan_policy, "high")
    low = _handle_nan_policy(low, nan_policy, "low")
    close = _handle_nan_policy(close, nan_policy, "close")
    # Ensure C-contiguous
    high = np.ascontiguousarray(high)
    low = np.ascontiguousarray(low)
    close = np.ascontiguousarray(close)
    # True Range
    tr = true_range_ind(high, low, close, drift)
    # Smooth TR with selected MA
    mamode = mamode.lower()
    if mamode == "rma":
        atr = rma_ind(tr, length, nan_policy=nan_policy)
    elif mamode == "sma":
        atr = sma_ind(tr, length, nan_policy=nan_policy)
    elif mamode == "ema":
        atr = ema_ind(tr, length, nan_policy=nan_policy)
    else:
        raise ValueError(f"Unsupported mamode: {mamode}")
    # Convert to percent if requested
    if percent:
        atr = atr * 100.0 / close
    # Trim if requested
    if trim:
        if len(atr) >= length:
            atr = atr[length - 1:]
        else:
            atr = np.array([])
    # Apply offset and fillna
    return _apply_offset_fillna(atr, offset, fillna)


# ----------------------------------------------------------------------
# ATR – TA-Lib wrapper with NaN handling and trim
# ----------------------------------------------------------------------
def atr_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """
    ATR using TA-Lib (C implementation) with pre‑processing.
    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    if length < 1:
        raise ValueError("length must be >= 1")

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    for name, arr in [("high", high), ("low", low), ("close", close)]:
        if np.isinf(arr).any():
            raise ValueError(f"Input {name} contains non-finite values (inf or -inf).")
    high = _handle_nan_policy(high, nan_policy, "high")
    low = _handle_nan_policy(low, nan_policy, "low")
    close = _handle_nan_policy(close, nan_policy, "close")

    high = np.ascontiguousarray(high)
    low = np.ascontiguousarray(low)
    close = np.ascontiguousarray(close)
    atr = talib.ATR(high, low, close, timeperiod=length)
    if percent:
        atr = atr * 100.0 / close
    if trim:
        if len(atr) >= length:
            atr = atr[length - 1:]
        else:
            atr = np.array([])
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
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
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
            high, low, close,
            length=length,
            offset=offset,
            fillna=fillna,
            percent=percent,
            nan_policy=nan_policy,
            trim=trim,
        )
    else:
        return atr_numba(
            high, low, close,
            length=length,
            mamode=mamode,
            drift=drift,
            offset=offset,
            fillna=fillna,
            percent=percent,
            nan_policy=nan_policy,
            trim=trim,
        )


# ----------------------------------------------------------------------
# Polars integration for ATR (always full length)
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
    nan_policy: str = 'raise',
    output_col: str | None = None
) -> pl.DataFrame:
    """
    ATR for Polars DataFrame (returns same length, no trim).
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
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars всегда возвращает полную длину
    )
    out_name = output_col or f"ATR_{length}"
    return df.with_columns([pl.Series(out_name, result)])