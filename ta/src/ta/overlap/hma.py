# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np
import polars as pl

from ..overlap import ema_ind, sma_ind, wma_ind
from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# HMA – Hull Moving Average (оптимизированная версия)
# ----------------------------------------------------------------------
def hma_numba(
    close: np.ndarray,
    length: int = 10,
    mamode: str = "wma",
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Hull Moving Average using Numba and selected base MA.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    ma_func: Callable[[np.ndarray, int], np.ndarray]
    if mamode == "sma":
        ma_func = sma_ind
    elif mamode == "ema":
        ma_func = ema_ind
    elif mamode == "wma":
        ma_func = wma_ind
    else:
        raise ValueError(f"Unsupported mamode: {mamode}")
    maf = ma_func(close, half_length)
    mas = ma_func(close, length)
    diff = 2.0 * maf - mas
    hma = ma_func(diff, sqrt_length)
    return _apply_offset_fillna(hma, offset, fillna)


# ----------------------------------------------------------------------
# Universal HMA function (accepts np.ndarray or pl.Series)
# ----------------------------------------------------------------------
def hma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    mamode: str = "wma",
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal Hull Moving Average (always uses Numba).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return hma_numba(close, length, mamode, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def hma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    mamode: str = "wma",
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    HMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length : int
        HMA period.
    mamode : str
        Type of moving average ('sma', 'ema', 'wma').
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default f"HMA_{length}").

    Returns
    -------
    pl.DataFrame
        HMA series.
    """
    close = df[close_col].to_numpy()
    result = hma_ind(close, length, mamode, offset, fillna)
    out_name = output_col or f"HMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])