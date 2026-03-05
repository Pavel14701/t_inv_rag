# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..overlap import sma_ind
from ..utils import _apply_offset_fillna
from . import atr_ind


def aberration_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 5,
    atr_length: int = 15,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy‑based Aberration calculation.

    Returns (zg, sg, xg, atr) as numpy arrays.
    """
    # Ensure contiguous
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # ATR (uses RMA by default)
    atr_arr = atr_ind(
        high, low, close,
        length=atr_length,
        mamode="rma",
        offset=0,
        fillna=None,
        percent=False,
        use_talib=use_talib,
    )
    # HLC3 and its SMA
    hlc3_arr = (high + low + close) / 3.0
    zg = sma_ind(hlc3_arr, length=length, offset=0, fillna=None, use_talib=use_talib)
    sg = zg + atr_arr
    xg = zg - atr_arr
    # Apply offset and fillna to all four lines
    zg = _apply_offset_fillna(zg, offset, fillna)
    sg = _apply_offset_fillna(sg, offset, fillna)
    xg = _apply_offset_fillna(xg, offset, fillna)
    atr_arr = _apply_offset_fillna(atr_arr, offset, fillna)
    return zg, sg, xg, atr_arr


def aberration_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 5,
    atr_length: int = 15,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal Aberration (accepts numpy arrays or Polars Series).
    Returns (zg, sg, xg, atr) as numpy arrays.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return aberration_numpy(
        high, low, close, length, atr_length, offset, fillna, use_talib
    )


def aberration_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    length: int = 5,
    atr_length: int = 15,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add Aberration columns to Polars DataFrame.

    Columns added:
        ABER_ZG_{length}_{atr_length}
        ABER_SG_{length}_{atr_length}
        ABER_XG_{length}_{atr_length}
        ABER_ATR_{length}_{atr_length}

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col, close_col : str
        Column names for prices.
    length : int
        Period for SMA of HLC3.
    atr_length : int
        Period for ATR.
    offset, fillna, use_talib : as usual.
    suffix : str
        Custom suffix (default f"_{length}_{atr_length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with four new columns.
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    zg, sg, xg, atr_arr = aberration_numpy(
        high, low, close, length, atr_length, offset, fillna, use_talib
    )
    suffix = suffix or f"_{length}_{atr_length}"
    return df.with_columns([
        pl.Series(f"ABER_ZG{suffix}", zg),
        pl.Series(f"ABER_SG{suffix}", sg),
        pl.Series(f"ABER_XG{suffix}", xg),
        pl.Series(f"ABER_ATR{suffix}", atr_arr),
    ])