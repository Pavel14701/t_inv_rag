# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _median_numba_core(close: np.ndarray, length: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    for i in range(length - 1, n):
        window = close[i - length + 1:i + 1].copy()
        # Используем partition для нахождения медианы без полной сортировки
        if length % 2 == 1:
            kth = length // 2
            # partition гарантирует, что элемент на позиции kth стоит на своём месте
            # (слева меньшие, справа большие)
            part = np.partition(window, kth)
            out[i] = part[kth]
        else:
            kth1 = length // 2 - 1
            kth2 = length // 2
            # Для чётной длины нужны два средних элемента
            part = np.partition(window, [kth1, kth2])
            out[i] = (part[kth1] + part[kth2]) * 0.5
    return out


def median_numba(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Rolling median using Numba (raw numpy version).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _median_numba_core(close, length)
    return _apply_offset_fillna(result, offset, fillna)


def median_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal rolling median (always uses Numba).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Window length.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        Median values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return median_numba(close, length, offset, fillna)


def median_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None,
) -> pl.Series:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length : int
        Window length.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default f"MEDIAN_{length}").

    Returns
    -------
    pl.Series
    """
    close = df[close_col].to_numpy()
    result = median_ind(close, length, offset, fillna)
    out_name = output_col or f"MEDIAN_{length}"
    return pl.Series(out_name, result)