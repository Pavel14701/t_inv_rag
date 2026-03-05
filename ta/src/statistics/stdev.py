# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@jit(nopython=True, fastmath=True, cache=True)
def _stdev_numba_core(close: np.ndarray, length: int, ddof: int) -> np.ndarray:
    """
    Скользящее стандартное отклонение через суммы и суммы квадратов (Numba).

    Параметры
    ---------
    close : np.ndarray
        Цены закрытия (float64).
    length : int
        Размер окна.
    ddof : int
        Delta Degrees of Freedom (0 или 1, для других ddof не тестировалось).

    Возвращает
    ----------
    np.ndarray
        Массив со значениями STDEV; первые length-1 элементов NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out

    # Накопительные суммы для первого окна
    sum_x = 0.0
    sum_x2 = 0.0
    for i in range(length):
        val = close[i]
        sum_x += val
        sum_x2 += val * val
    # Вычисляем std для первого окна
    mean = sum_x / length
    variance = (sum_x2 - 2 * mean * sum_x + length * mean * mean) / (length - ddof)
    out[length - 1] = np.sqrt(variance) if variance >= 0 else np.nan
    # Скользящее обновление сумм
    for i in range(length, n):
        # Добавляем новый элемент, удаляем самый старый
        new_val = close[i]
        old_val = close[i - length]
        sum_x += new_val - old_val
        sum_x2 += new_val * new_val - old_val * old_val
        # Пересчитываем std
        mean = sum_x / length
        variance = (sum_x2 - 2 * mean * sum_x + length * mean * mean) / (length - ddof)
        out[i] = np.sqrt(variance) if variance >= 0 else np.nan
    return out


def stdev_numba(
    close: np.ndarray,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> np.ndarray:
    """
    Скользящее стандартное отклонение через Numba (чистая версия).
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = _stdev_numba_core(close, length, ddof)
    return _apply_offset_fillna(result, offset, fillna)


def stdev_talib(
    close: np.ndarray,
    length: int = 30,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> np.ndarray:
    """
    Скользящее стандартное отклонение через TA-Lib (ddof=0).
    """
    if not talib_available:
        raise ImportError("TA-Lib not available")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    result = talib.STDDEV(close, timeperiod=length)
    return _apply_offset_fillna(result, offset, fillna)


def stdev_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Универсальная функция скользящего стандартного отклонения.

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
        Массив со значениями STDEV.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return stdev_talib(close, length, offset, fillna)
    else:
        return stdev_numba(close, length, ddof, offset, fillna)


def stdev_polars(
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
    Добавляет колонку со скользящим стандартным отклонением в Polars DataFrame.
    """
    close = df[close_col].to_numpy()
    result = stdev_ind(
        close,
        length=length,
        ddof=ddof,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )
    out_name = output_col or f"STDEV_{length}"
    return pl.Series(out_name, result)


def stdev_polars_multi(
    df: pl.DataFrame,
    columns: list[str],
    length: int = 30,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    suffix: str = "_stdev",
) -> pl.DataFrame:
    """
    Добавляет колонки со скользящим стандартным отклонением для нескольких колонок,
    используя параллельные возможности Polars.

    Параметры
    ---------
    df : pl.DataFrame
        Исходные данные.
    columns : list[str]
        Список колонок, для которых нужно вычислить STDEV.
    length : int
        Период окна.
    ddof : int
        Delta Degrees of Freedom (для Polars rolling_std всегда использует ddof=1,
        но параметр сохранён для совместимости).
    offset : int
        Сдвиг результата (положительный – вперёд).
    fillna : float, optional
        Значение для заполнения NaN после сдвига.
    suffix : str
        Суффикс, добавляемый к исходному имени колонки для формирования выходной.

    Возвращает
    ----------
    pl.DataFrame
        Исходный DataFrame с новыми колонками вида `{col}{suffix}`.
    """
    exprs = [
        pl.col(col).rolling_std(window_size=length, ddof=ddof).alias(f"{col}{suffix}")
        for col in columns
    ]
    df = df.with_columns(exprs)
    if offset != 0:
        shift_exprs = [
            pl.col(f"{col}{suffix}").shift(offset).alias(f"{col}{suffix}")
            for col in columns
        ]
        df = df.with_columns(shift_exprs)
    if fillna is not None:
        fill_exprs = [
            pl.col(f"{col}{suffix}").fill_nan(fillna).alias(f"{col}{suffix}")
            for col in columns
        ]
        df = df.with_columns(fill_exprs)
    return df