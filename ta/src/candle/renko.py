# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int8, njit

from ..utils import _apply_offset_fillna


@njit(
    (float64[:], float64),
    nopython=True,
    cache=True
)
def _renko_nb(
    prices: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    Numba-accelerated Renko brick stream aligned to bars.
    Returns int8 array:
        +1 → up brick
        -1 → down brick
         0 → no brick
    """
    n = prices.size
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    # start from first price
    anchor = prices[0]

    for i in range(1, n):
        p = prices[i]

        # сколько box-ов прошло вверх
        while p >= anchor + box_size:
            anchor += box_size
            out[i] = 1

        # сколько box-ов прошло вниз
        while p <= anchor - box_size:
            anchor -= box_size
            out[i] = -1

    return out


def renko(
    prices: np.ndarray | pl.Series,
    box_size: float,
    offset: int = 0,
    fillna: float | None = None,
) -> np.ndarray:
    """
    Universal Renko brick stream.
    Returns float64 array aligned to bars:
        +1.0 → up brick
        -1.0 → down brick
         0.0 → no brick
    """
    if isinstance(prices, pl.Series):
        prices = prices.to_numpy()

    prices = np.asarray(prices, dtype=np.float64)
    if not prices.flags.c_contiguous:
        prices = np.ascontiguousarray(prices)

    bricks = _renko_nb(prices, float(box_size))
    out = bricks.astype(np.float64)
    return _apply_offset_fillna(out, offset, fillna)


def renko_polars(
    df: pl.DataFrame,
    price_col: str = "close",
    box_size: float = 1.0,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str = "RENKO",
) -> pl.DataFrame:
    """
    Add Renko brick stream to Polars DataFrame.
    """
    out = renko(
        df[price_col].to_numpy(),
        box_size=box_size,
        offset=offset,
        fillna=fillna,
    )
    return df.with_columns(pl.Series(output_col, out))
