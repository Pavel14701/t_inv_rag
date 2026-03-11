# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, njit

from ..utils import _apply_offset_fillna


@njit(
    (float64[:], float64),
    nopython=True,
    cache=True
)
def _kagi_nb(
    prices: np.ndarray,
    reversal: float,
) -> np.ndarray:
    """
    Numba-accelerated Kagi line (yin/yang) aligned to bars.
    Returns int8 array:
        +1 → yang (up)
        -1 → yin (down)
         0 → not formed yet
    """
    n = prices.size
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out
    p0 = prices[0]
    direction = 0  # 1=up, -1=down
    last_extreme = p0
    for i in range(1, n):
        p = prices[i]
        if direction == 0:
            # ищем первое направление
            if p >= p0 + reversal:
                direction = 1
                last_extreme = p
            elif p <= p0 - reversal:
                direction = -1
                last_extreme = p
            out[i] = direction
            continue
        if direction == 1:
            # продолжаем вверх
            if p > last_extreme:
                last_extreme = p
                out[i] = 1
                continue
            # разворот вниз
            if p <= last_extreme - reversal:
                direction = -1
                last_extreme = p
                out[i] = -1
                continue
            out[i] = 1
        else:  # direction == -1
            # продолжаем вниз
            if p < last_extreme:
                last_extreme = p
                out[i] = -1
                continue
            # разворот вверх
            if p >= last_extreme + reversal:
                direction = 1
                last_extreme = p
                out[i] = 1
                continue

            out[i] = -1

    return out


def kagi(
    prices: np.ndarray | pl.Series,
    reversal: float,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal Kagi yin/yang stream.
    Returns float64 array aligned to bars:
        +1.0 → yang (up)
        -1.0 → yin (down)
         0.0 → not formed yet
    """
    if isinstance(prices, pl.Series):
        prices = prices.to_numpy()
    prices = np.asarray(prices, dtype=np.float64)
    if not prices.flags.c_contiguous:
        prices = np.ascontiguousarray(prices)
    arr = _kagi_nb(prices, reversal)
    out = arr.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def kagi_polars(
    df: pl.DataFrame,
    price_col: str = "close",
    reversal: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "KAGI",
) -> pl.DataFrame:
    """
    Add Kagi yin/yang stream to Polars DataFrame.
    """
    out = kagi(
        df[price_col].to_numpy(),
        reversal=reversal,
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
