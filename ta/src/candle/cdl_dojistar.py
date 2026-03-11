# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import njit, types

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@njit(
    (
        types.float64[:], types.float64[:], types.float64[:], types.float64[:],
        types.float64, types.float64, types.boolean, types.boolean
    ),
    nopython=True,
    cache=True,
    fastmath=True,
)
def _cdl_dojistar_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_body_factor: float,
    max_shadow_factor: float,
    strict: bool,
    symmetric: bool,  # kept for API consistency
) -> np.ndarray:
    """
    Optimized Doji Star pattern.

    Returns:
        1.0  → bullish dojistar
       -1.0  → bearish dojistar
        0.0  → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        o1 = open_[i - 1]
        c1 = close[i - 1]
        h1 = high[i - 1]
        l1 = low[i - 1]

        o0 = open_[i]
        c0 = close[i]
        h0 = high[i]
        l0 = low[i]

        r1 = h1 - l1
        r0 = h0 - l0
        if r1 <= 0.0 or r0 <= 0.0:
            continue

        # Doji body
        body0 = c0 - o0 if c0 > o0 else o0 - c0
        if body0 > min_body_factor * r0:
            continue

        # Gap logic
        bull = False
        bear = False

        # Bullish dojistar: gap down
        if h0 < l1:
            bull = True

        # Bearish dojistar: gap up
        elif l0 > h1:
            bear = True

        if not (bull or bear):
            continue

        if strict:
            # Shadows (fast)
            up0 = o0 if o0 > c0 else c0
            lo0 = c0 if o0 > c0 else o0
            sh0 = (h0 - up0) + (lo0 - l0)

            if sh0 > max_shadow_factor * r0:
                continue

        out[i] = 1.0 if bull else -1.0

    return out


def cdl_dojistar(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    strict: bool = False,
    symmetric: bool = False,
    min_body_factor: float = 0.1,
    max_shadow_factor: float = 1.0,
) -> np.ndarray:
    """
    Doji Star pattern with strict support.
    """
    if isinstance(open_, pl.Series): open_ = open_.to_numpy()
    if isinstance(high, pl.Series): high = high.to_numpy()
    if isinstance(low, pl.Series): low = low.to_numpy()
    if isinstance(close, pl.Series): close = close.to_numpy()

    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if use_talib and talib_available and not symmetric:
        talib_out = talib.CDLDOJISTAR(open_, high, low, close)
        result = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(result, offset, fillna)

    out = _cdl_dojistar_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric,
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_dojistar_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    strict: bool = False,
    symmetric: bool = False,
    min_body_factor: float = 0.1,
    max_shadow_factor: float = 1.0,
    output_col: str = "CDL_DOJISTAR",
) -> pl.DataFrame:
    out = cdl_dojistar(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
        strict=strict,
        symmetric=symmetric,
        min_body_factor=min_body_factor,
        max_shadow_factor=max_shadow_factor,
    )
    return df.with_columns(pl.Series(output_col, out))
