# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..overlap import wma_ind
from ..utils import _apply_offset_fillna
from . import roc_ind


def coppock_numpy(
    close: np.ndarray,
    fast: int = 11,
    slow: int = 14,
    wma_length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based Coppock Curve calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    fast, slow : int
        ROC periods.
    wma_length : int
        WMA smoothing period.
    offset, fillna, use_talib : as usual.

    Returns
    -------
    np.ndarray
        Coppock values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    roc_fast = roc_ind(close, length=fast, use_talib=use_talib)
    roc_slow = roc_ind(close, length=slow, use_talib=use_talib)
    total_roc = roc_fast + roc_slow
    coppock = wma_ind(total_roc, length=wma_length, use_talib=use_talib)
    return _apply_offset_fillna(coppock, offset, fillna)


def coppock_ind(
    close: np.ndarray | pl.Series,
    fast: int = 11,
    slow: int = 14,
    wma_length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Coppock Curve (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return coppock_numpy(close, fast, slow, wma_length, offset, fillna, use_talib)


def coppock_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    fast: int = 11,
    slow: int = 14,
    wma_length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    fast, slow, wma_length, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"COPC_{fast}_{slow}_{wma_length}").

    Returns
    -------
    pl.DataFrame
    """
    close = df[close_col].to_numpy()
    result = coppock_numpy(close, fast, slow, wma_length, offset, fillna, use_talib)
    out_name = output_col or f"COPC_{fast}_{slow}_{wma_length}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })