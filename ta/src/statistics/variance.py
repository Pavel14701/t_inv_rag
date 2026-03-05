# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _variance_numba_core(close: np.ndarray, length: int, ddof: int) -> np.ndarray:
    """
    Скользящая дисперсия через суммы и суммы квадратов (Numba).

    Параметры
    ---------
    close : np.ndarray
        Цены закрытия (float64).
    length : int
        Размер окна.
    ddof : int
        Delta Degrees of Freedom (0 или 1).

    Возвращает
    ----------
    np.ndarray
        Массив со значениями дисперсии; первые length-1 элементов NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(length):
        val = close[i]
        sum_x += val
        sum_x2 += val * val
    mean = sum_x / length
    variance = (sum_x2 - 2 * mean * sum_x + length * mean * mean) / (length - ddof)
    out[length - 1] = variance if variance >= 0 else np.nan
    for i in range(length, n):
        new_val = close[i]
        old_val = close[i - length]
        sum_x += new_val - old_val
        sum_x2 += new_val * new_val - old_val * old_val
        mean = sum_x / length
        variance = (sum_x2 - 2 * mean * sum_x + length * mean * mean) / (length - ddof)
        out[i] = variance if variance >= 0 else np.nan
    return out


def variance_numba(
    close: np.ndarray,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> np.ndarray:
    """
    Скользящая дисперсия через Numba (чистая версия).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _variance_numba_core(close, length, ddof)
    return _apply_offset_fillna(result, offset, fillna)


def variance_talib(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> np.ndarray:
    """
    Скользящая дисперсия через TA-Lib (ddof=0).
    """
    if not talib_available:
        raise ImportError("TA-Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = talib.VAR(close, timeperiod=length)
    return _apply_offset_fillna(result, offset, fillna)


def variance_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Универсальная функция скользящей дисперсии.

    Параметры
    ---------
    close : np.ndarray или pl.Series
        Цены закрытия.
    length : int
        Период.
    ddof : int
        Delta Degrees of Freedom. Для TA-Lib всегда 0.
    offset : int
        Сдвиг результата.
    fillna : float, optional
        Значение для заполнения NaN.
    use_talib : bool
        Если True и TA-Lib доступна, использует её (ddof=0).

    Возвращает
    ----------
    np.ndarray
        Массив со значениями дисперсии.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return variance_talib(close, length, offset, fillna)
    else:
        return variance_numba(close, length, ddof, offset, fillna)


def variance_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    output_col: Optional[str] = None,
) -> pl.Series:
    """
    Добавляет колонку со скользящей дисперсией в Polars DataFrame.
    """
    close = df[close_col].to_numpy()
    result = variance_ind(
        close,
        length=length,
        ddof=ddof,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )
    out_name = output_col or f"VAR_{length}"
    return pl.Series(out_name, result)