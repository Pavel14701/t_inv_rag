# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib_available
from ..overlap import rma_ind
from ..utils import _apply_offset_fillna

if talib_available:
    import talib


def rsi_numpy(
    close: np.ndarray,
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based RSI calculation.

    Returns RSI values as numpy array.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if use_talib and talib_available:
        rsi = talib.RSI(close, timeperiod=length)
    else:
        # Compute differences with given drift
        n = len(close)
        diff = np.full(n, np.nan, dtype=np.float64)
        for i in range(drift, n):
            diff[i] = close[i] - close[i - drift]
        gain = np.maximum(diff, 0.0)
        loss = np.maximum(-diff, 0.0)
        avg_gain = rma_ind(gain, length, offset=0, fillna=None)
        avg_loss = rma_ind(loss, length, offset=0, fillna=None)
        rs = avg_gain / avg_loss
        rsi = scalar - scalar / (1.0 + rs)
    return _apply_offset_fillna(rsi, offset, fillna)


def rsi_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal RSI (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return rsi_numpy(close, length, scalar, drift, offset, fillna, use_talib)


def rsi_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    length: int = 14,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add RSI column to Polars DataFrame.
    """
    close = df[close_col].to_numpy()
    result = rsi_numpy(close, length, scalar, drift, offset, fillna, use_talib)
    out_name = output_col or f"RSI_{length}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })