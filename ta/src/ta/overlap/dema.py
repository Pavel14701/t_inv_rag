# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..utils import _apply_offset_fillna
from .ema import _ema_numba_opt


def dema_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """Double Exponential Moving Average using Numba (fallback)."""
    # ensure float64 without unnecessary copy
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    ema1 = _ema_numba_opt(close, length)
    ema2 = _ema_numba_opt(ema1, length)
    dema = 2.0 * ema1 - ema2
    # apply offset and fillna in one optimized pass
    return _apply_offset_fillna(dema, offset, fillna)


def dema_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """DEMA via TA-Lib (primary when available)."""
    if not talib:
        raise ImportError("TA-Lib is not available")
    close = close.astype(np.float64)
    dema = talib.DEMA(close, timeperiod=length)
    if offset != 0:
        dema = np.roll(dema, offset)
        if offset > 0:
            dema[:offset] = np.nan
        else:
            dema[offset:] = np.nan
    if fillna is not None:
        dema = np.where(np.isnan(dema), fillna, dema)
    return dema


def dema_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> np.ndarray:
    """Universal DEMA with automatic implementation selection."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = close.astype(np.float64)
    if use_talib and talib_available:
        return dema_talib(close, length, offset, fillna)
    else:
        return dema_numba(close, length, offset, fillna)


def dema_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None
) -> pl.DataFrame:
    """Wrapper for Polars DataFrame."""
    close = df[close_col].to_numpy()
    result = dema_ind(
        close, 
        length=length, 
        offset=offset, 
        fillna=fillna, 
        use_talib=use_talib
    )
    output_name = output_col or f"DEMA_{length}"
    return df.with_columns([pl.Series(output_name, result)])