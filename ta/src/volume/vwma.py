# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..overlap import sma_ind
from ..utils import _apply_offset_fillna


def vwma_numpy(
    close: np.ndarray,
    volume: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based VWMA calculation.

    Parameters
    ----------
    close, volume : np.ndarray
        Price and volume arrays (float64), same length.
    length : int
        SMA period.
    offset, fillna, use_talib : as usual.

    Returns
    -------
    np.ndarray
        VWMA values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    volume = np.asarray(volume, dtype=np.float64, copy=False)
    for arr in (close, volume):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # Price * volume
    pv = close * volume
    # SMA of pv and volume
    sma_pv = sma_ind(pv, length=length, offset=0, fillna=None, use_talib=use_talib)
    sma_vol = sma_ind(volume, length=length, offset=0, fillna=None, use_talib=use_talib)
    # VWMA = SMA(pv) / SMA(vol)
    with np.errstate(divide='ignore', invalid='ignore'):
        vwma = sma_pv / sma_vol
    return _apply_offset_fillna(vwma, offset, fillna)


def vwma_ind(
    close: np.ndarray | pl.Series,
    volume: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal VWMA (accepts numpy arrays or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if isinstance(volume, pl.Series):
        volume = volume.to_numpy()
    return vwma_numpy(close, volume, length, offset, fillna, use_talib)


def vwma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add VWMA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col, volume_col : str
        Column names for prices and volume.
    length, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"VWMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    close = df[close_col].to_numpy()
    volume = df[volume_col].to_numpy()
    result = vwma_numpy(close, volume, length, offset, fillna, use_talib)
    out_name = output_col or f"VWMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])