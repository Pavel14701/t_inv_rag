# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, int8, njit

from ..utils import _apply_offset_fillna


@njit(
    (float64[:], float64, int64),
    nopython=True,
    cache=True
)
def _pf_trend_nb(
    prices: np.ndarray,
    box_size: float,
    reversal: int
) -> np.ndarray:
    """
    Numba-accelerated Point & Figure trend state (X/O) by close prices.

    Returns int8 array:
    -  1  → колонка X (бычья)
    - -1  → колонка O (медвежья)
    -  0  → ещё нет сформированной колонки
    """
    n = prices.size
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    p0 = prices[0]
    cur_kind = 0  # 0 = none, 1 = X, -1 = O
    col_top = p0
    col_bottom = p0

    # ищем первую колонку
    for i in range(1, n):
        p = prices[i]
        if cur_kind == 0:
            if p >= p0 + box_size:
                cur_kind = 1
                col_bottom = np.floor(p0 / box_size) * box_size
                col_top = np.floor(p / box_size) * box_size
            elif p <= p0 - box_size:
                cur_kind = -1
                col_top = np.floor(p0 / box_size) * box_size
                col_bottom = np.floor(p / box_size) * box_size
            out[i] = cur_kind
            continue

        if cur_kind == 1:
            # продолжаем X вверх
            needed_up = col_top + box_size
            if p >= needed_up:
                col_top = np.floor(p / box_size) * box_size
                out[i] = 1
                continue

            # разворот в O
            rev_level = col_top - box_size * reversal
            if p <= rev_level:
                cur_kind = -1
                col_bottom = np.floor(p / box_size) * box_size
                out[i] = -1
                continue

            out[i] = 1
        else:
            # cur_kind == -1
            # продолжаем O вниз
            needed_down = col_bottom - box_size
            if p <= needed_down:
                col_bottom = np.floor(p / box_size) * box_size
                out[i] = -1
                continue

            # разворот в X
            rev_level = col_bottom + box_size * reversal
            if p >= rev_level:
                cur_kind = 1
                col_top = np.floor(p / box_size) * box_size
                out[i] = 1
                continue

            out[i] = -1

    return out


def pf_trend(
    prices: np.ndarray | pl.Series,
    box_size: float,
    reversal: int = 3,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Point & Figure trend state по ряду цен (обычно close).

    Возвращает float64-массив:
    -  1.0 → колонка X (рост)
    - -1.0 → колонка O (падение)
    -  0.0 → колонка ещё не сформирована
    """
    if isinstance(prices, pl.Series):
        prices = prices.to_numpy()

    prices = np.asarray(prices, dtype=np.float64)
    if not prices.flags.c_contiguous:
        prices = np.ascontiguousarray(prices)

    trend_int = _pf_trend_nb(prices, float(box_size), int(reversal))
    out = trend_int.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def pf_trend_polars(
    df: pl.DataFrame,
    price_col: str = "close",
    box_size: float = 1.0,
    reversal: int = 3,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "PF_TREND",
) -> pl.DataFrame:
    """
    Добавляет колонку Point & Figure тренда в Polars DataFrame.
    """
    out = pf_trend(
        df[price_col].to_numpy(),
        box_size=box_size,
        reversal=reversal,
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
