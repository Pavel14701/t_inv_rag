from typing import Literal, Optional, Union

import numpy as np
import polars as pl
from numba import jit

# ------------------------------------------------------------
# Numba-функции для скользящих средних
# ------------------------------------------------------------


@jit(nopython=True)
def _sma_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее (SMA) через Numba."""
    n = len(arr)
    res = np.full(n, np.nan)
    if n < window:
        return res
    cumsum = np.cumsum(arr)
    res[window - 1:] = (
        cumsum[window - 1:] - np.concatenate(
            ([0], cumsum[:n - window])
        )
    ) / window
    return res


@jit(nopython=True)
def _ema_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Экспоненциальное скользящее среднее (EMA) через Numba."""
    n = len(arr)
    res = np.full(n, np.nan)
    if n < window:
        return res
    alpha = 2.0 / (window + 1)
    res[window - 1] = np.mean(arr[:window])  # стартовое значение – SMA
    for i in range(window, n):
        res[i] = alpha * arr[i] + (1 - alpha) * res[i - 1]
    return res


# Словарь доступных скользящих средних
_MA_FUNCTIONS = {
    "sma": _sma_numba,
    "ema": _ema_numba,
}

# ------------------------------------------------------------
# Основная функция accbands
# ------------------------------------------------------------


def accbands(
    high: Union[np.ndarray, pl.Series],
    low: Union[np.ndarray, pl.Series],
    close: Union[np.ndarray, pl.Series],
    length: int = 20,
    c: float = 4.0,
    mamode: Literal["sma", "ema"] = "sma",
    drift: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
) -> pl.DataFrame:
    """
    Acceleration Bands (ACCBANDS) – полный аналог pandas_ta.accbands,
    реализованный на NumPy + Numba, возвращает Polars DataFrame.

    Параметры
    ---------
    high, low, close : np.ndarray или pl.Series
        Входные временные ряды (цены). Должны быть одинаковой длины.
    length : int
        Период скользящего среднего (по умолчанию 20).
    c : float
        Множитель полос (по умолчанию 4.0).
    mamode : {'sma', 'ema'}
        Тип скользящего среднего (по умолчанию 'sma').
    drift : int
        Не используется в расчётах, оставлен для совместимости.
    offset : int
        Сдвиг результирующих рядов (положительное значение – вперёд, 
        отрицательное – назад).
    fillna : float, optional
        Значение для заполнения пропусков (NaN) в результате.

    Возвращает
    ----------
    pl.DataFrame
        С тремя колонками: 'ACCBL_{length}', 'ACCBM_{length}', 'ACCBU_{length}'.
    """
    # 1. Преобразование входных данных в numpy массивы (если пришли Polars Series)
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Проверка на одинаковую длину
    n = len(high)
    if not (len(low) == len(close) == n):
        raise ValueError("high, low, close должны быть одной длины")

    # 2. Валидация параметров
    if length <= 0:
        raise ValueError("length должен быть положительным")
    if c <= 0:
        raise ValueError("c должен быть положительным")
    if mamode not in _MA_FUNCTIONS:
        raise ValueError(f"mamode должен быть одним из {list(_MA_FUNCTIONS.keys())}")

    # 3. Выбор функции скользящего среднего
    ma_func = _MA_FUNCTIONS[mamode]

    # 4. Расчёт вспомогательных величин
    high_low_range = np.maximum(high - low, 0.0)            # non_zero_range
    hl_ratio = high_low_range / (high + low) * c            # (range/(high+low)) * c
    # защита от деления на ноль: если high+low == 0, то ratio будет NaN, что допустимо

    _lower = low * (1 - hl_ratio)
    _upper = high * (1 + hl_ratio)

    # 5. Расчёт скользящих средних (основные полосы)
    lower = ma_func(_lower, length)
    mid = ma_func(close, length)
    upper = ma_func(_upper, length)

    # 6. Применение сдвига (offset)
    if offset != 0:
        lower = np.roll(lower, offset)
        mid = np.roll(mid, offset)
        upper = np.roll(upper, offset)
        # Заполняем образовавшиеся края NaN
        if offset > 0:
            lower[:offset] = np.nan
            mid[:offset] = np.nan
            upper[:offset] = np.nan
        elif offset < 0:
            lower[offset:] = np.nan
            mid[offset:] = np.nan
            upper[offset:] = np.nan

    # 7. Заполнение пропусков (fillna)
    if fillna is not None:
        lower = np.where(np.isnan(lower), fillna, lower)
        mid = np.where(np.isnan(mid), fillna, mid)
        upper = np.where(np.isnan(upper), fillna, upper)

    # 8. Формирование Polars DataFrame
    result = pl.DataFrame({
        f"ACCBL_{length}": lower,
        f"ACCBM_{length}": mid,
        f"ACCBU_{length}": upper,
    })

    return result


# ------------------------------------------------------------
# Пример использования
# ------------------------------------------------------------
if __name__ == "__main__":
    # Генерируем тестовые данные
    np.random.seed(42)
    n = 100
    high = np.random.randn(n).cumsum() + 100
    low = high - np.abs(np.random.randn(n) * 0.5)
    close = (high + low) / 2 + np.random.randn(n) * 0.2

    # Вызов функции
    df_acc = accbands(high, low, close, length=20, c=4.0, mamode="sma")
    print(df_acc)