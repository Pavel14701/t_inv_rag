# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna
from . import ema_ind, sma_ind


def ma_numba(arr: np.ndarray, length: int, mamode: str = "sma") -> np.ndarray:
    """Unified moving average using Numba (supports 'sma', 'ema')."""
    arr = arr.astype(np.float64)
    if mamode.lower() == "sma":
        return sma_ind(arr, length)
    elif mamode.lower() == "ema":
        return ema_ind(arr, length)
    else:
        raise ValueError(f"Unsupported mamode: {mamode}")


def ma_talib(arr: np.ndarray, length: int, mamode: str = "sma") -> np.ndarray:
    """Unified moving average using TA-Lib."""
    if not talib_available:
        raise ImportError("TA-Lib not available")
    arr = arr.astype(np.float64)
    mamode = mamode.lower()
    if mamode == "sma":
        return talib.SMA(arr, timeperiod=length)
    elif mamode == "ema":
        return talib.EMA(arr, timeperiod=length)
    else:
        raise ValueError(f"Unsupported mamode: {mamode}")


def ma(
    arr: np.ndarray,
    length: int,
    mamode: str = "sma",
    use_talib: bool = True
) -> np.ndarray:
    """Universal moving average with automatic backend selection."""
    if use_talib and talib_available:
        return ma_talib(arr, length, mamode)
    else:
        return ma_numba(arr, length, mamode)


# ----------------------------------------------------------------------
# Основная логика HiLo Activator (Numba-цикл)
# ----------------------------------------------------------------------

@jit(nopython=True, fastmath=True, cache=True)
def _hilo_numba_core(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    high_ma: np.ndarray,
    low_ma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-ядро для HiLo Activator.
    Возвращает три массива: hilo, long, short.
    """
    n = len(close)
    hilo = np.full(n, np.nan, dtype=np.float64)
    long = np.full(n, np.nan, dtype=np.float64)
    short = np.full(n, np.nan, dtype=np.float64)
    # Первый элемент неопределён, начинаем с i=1
    if n < 2:
        return hilo, long, short
    # Инициализация первого значения (как в оригинале: берём предыдущее значение MA?)
    # Оригинал: цикл с i=1, проверка close[i] > high_ma[i-1] и т.д.
    # Для i=0 все значения остаются NaN (так же как в pandas_ta)
    for i in range(1, n):
        if close[i] > high_ma[i - 1]:
            hilo[i] = low_ma[i]
            long[i] = low_ma[i]
            short[i] = np.nan
        elif close[i] < low_ma[i - 1]:
            hilo[i] = high_ma[i]
            short[i] = high_ma[i]
            long[i] = np.nan
        else:
            hilo[i] = hilo[i - 1]
            long[i] = hilo[i - 1]
            short[i] = hilo[i - 1]
    return hilo, long, short


def _hilo_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HiLo Activator полностью на Numba (включая MA).
    Возвращает (hilo, long, short).
    """
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    close = close.astype(np.float64)
    # Расчёт скользящих средних
    high_ma = ma_numba(high, high_length, mamode)
    low_ma = ma_numba(low, low_length, mamode)
    hilo, long_arr, short_arr = _hilo_numba_core(high, low, close, high_ma, low_ma)
    # Применяем offset и fillna единым образом к каждому массиву
    hilo = _apply_offset_fillna(hilo, offset, fillna)
    long_arr = _apply_offset_fillna(long_arr, offset, fillna)
    short_arr = _apply_offset_fillna(short_arr, offset, fillna)
    return hilo, long_arr, short_arr


def _hilo_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HiLo Activator с использованием TA-Lib для MA и Numba для логики.
    """
    if not talib_available:
        raise ImportError("TA-Lib not available")
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    close = close.astype(np.float64)
    # MA через TA-Lib
    high_ma = ma_talib(high, high_length, mamode)
    low_ma = ma_talib(low, low_length, mamode)
    hilo, long_arr, short_arr = _hilo_numba_core(high, low, close, high_ma, low_ma)
    # Применяем offset и fillna единым образом к каждому массиву
    hilo = _apply_offset_fillna(hilo, offset, fillna)
    long_arr = _apply_offset_fillna(long_arr, offset, fillna)
    short_arr = _apply_offset_fillna(short_arr, offset, fillna)
    return hilo, long_arr, short_arr


def hilo_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Универсальная функция HiLo Activator.
    Возвращает кортеж из трёх numpy массивов (hilo, long, short).
    """
    # Преобразование Polars Series в numpy
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if use_talib and talib_available:
        return _hilo_talib(
            high, low, close, high_length, low_length, mamode, offset, fillna
            )
    else:
        return _hilo_numba(
            high, low, close, high_length, low_length, mamode, offset, fillna
            )


# ----------------------------------------------------------------------
# Обёртки для Polars DataFrame
# ----------------------------------------------------------------------
def hilo_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    high_length: int = 13,
    low_length: int = 21,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = ""
) -> pl.DataFrame:
    """
    Adds columns HILO, HILOl, HILOs to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Original DataFrame.
    high_col, low_col, close_col : str
        Names colomns with prices.
    high_length, low_length : int
        Periods for moving averages.
    mamode : str
        Type of moving average ('sma', 'ema').
    offset : int
        Result's offset.
    fillna : float, optional
        Fill value NaN.
    use_talib : bool
        Use TA-Lib for MA, if available.
    suffix : str
        Suffix for output column names (e.g. "_fast").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added columns.
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    hilo_arr, long_arr, short_arr = hilo_ind(
        high, low, close,
        high_length=high_length,
        low_length=low_length,
        mamode=mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib
    )
    suffix = suffix or f"_{high_length}_{low_length}"
    return df.with_columns([
        pl.Series(f"HILO{suffix}", hilo_arr),
        pl.Series(f"HILOl{suffix}", long_arr),
        pl.Series(f"HILOs{suffix}", short_arr)
    ])