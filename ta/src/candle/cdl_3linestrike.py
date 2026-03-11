# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import njit, types

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@njit(
    (types.float64[:], types.float64[:], types.float64[:], types.float64[:],
     types.float64, types.float64, types.boolean),
    nopython=True,
    cache=True,
    fastmath=True
)
def _cdl_3linestrike_nb(
    open_, high, low, close,
    min_body_factor,
    max_shadow_factor,
    strict
):
    """
    Numba‑accelerated Three‑Line Strike pattern with optional strict filtering.

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        OHLC prices (float64).
    min_body_factor : float
        Minimum body size as a fraction of the total range (0 = disabled).
    max_shadow_factor : float
        Maximum total shadow (upper+lower) as a fraction of total range (1 = disabled).
    strict : bool
        If True, apply additional filters (min body, max shadow). \
            Otherwise only basic pattern.

    Returns
    -------
    np.ndarray
        1.0 (bullish), -1.0 (bearish), 0.0 (none).
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)
    for i in range(3, n):
        # --- 4 candles OHLC ---
        o3, c3 = open_[i - 3], close[i - 3]
        o2, c2 = open_[i - 2], close[i - 2]
        o1, c1 = open_[i - 1], close[i - 1]
        o0, c0 = open_[i], close[i]
        h3, l3 = high[i - 3], low[i - 3]
        h2, l2 = high[i - 2], low[i - 2]
        h1, l1 = high[i - 1], low[i - 1]
        h0, l0 = high[i], low[i]
        # --- basic pattern (без strict) ---
        # три белых, каждый выше предыдущего
        bull3 = (
            (c3 > o3) and (c2 > o2) and (c1 > o1) and
            (c2 > c3) and (c1 > c2)
        )
        # три чёрных, каждый ниже предыдущего
        bear3 = (
            (c3 < o3) and (c2 < o2) and (c1 < o1) and
            (c2 < c3) and (c1 < c2)
        )
        direction = 0.0
        if bull3:
            # чёрная ударная, открытие выше close третьей, закрытие ниже close первой
            if (c0 < o0) and (o0 > c1) and (c0 < c3):
                direction = 1.0
            else:
                continue
        elif bear3:
            # белая ударная, открытие ниже close третьей, закрытие выше close первой
            if (c0 > o0) and (o0 < c1) and (c0 > c3):
                direction = -1.0
            else:
                continue
        else:
            continue
        # --- strict‑фильтры одним блоком ---
        if strict:
            # диапазоны
            r3 = h3 - l3
            r2 = h2 - l2
            r1 = h1 - l1
            r0 = h0 - l0
            # тела
            b3 = abs(c3 - o3)
            b2 = abs(c2 - o2)
            b1 = abs(c1 - o1)
            b0 = abs(c0 - o0)
            if (b3 < min_body_factor * r3 or
                b2 < min_body_factor * r2 or
                b1 < min_body_factor * r1 or
                b0 < min_body_factor * r0):
                continue

            # тени без max/min
            up3 = o3 if o3 > c3 else c3
            lo3 = c3 if o3 > c3 else o3
            sh3 = (h3 - up3) + (lo3 - l3)

            up2 = o2 if o2 > c2 else c2
            lo2 = c2 if o2 > c2 else o2
            sh2 = (h2 - up2) + (lo2 - l2)

            up1 = o1 if o1 > c1 else c1
            lo1 = c1 if o1 > c1 else o1
            sh1 = (h1 - up1) + (lo1 - l1)

            up0 = o0 if o0 > c0 else c0
            lo0 = c0 if o0 > c0 else o0
            sh0 = (h0 - up0) + (lo0 - l0)

            if (sh3 > max_shadow_factor * r3 or
                sh2 > max_shadow_factor * r2 or
                sh1 > max_shadow_factor * r1 or
                sh0 > max_shadow_factor * r0):
                continue

        out[i] = direction

    return out


def cdl_3linestrike(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    strict: bool = False,
    min_body_factor: float = 0.0,
    max_shadow_factor: float = 1.0,
) -> np.ndarray:
    """
    Universal Three‑Line Strike pattern with optional strict mode.

    Parameters
    ----------
    open_, high, low, close : array-like
        OHLC prices.
    offset, fillna, use_talib : as usual.
    strict : bool
        If True, apply additional filters (min body size, max shadow).
    min_body_factor : float
        Minimum body size as fraction of candle range (only if strict=True).
    max_shadow_factor : float
        Maximum total shadow as fraction of candle range (only if strict=True).

    Returns
    -------
    np.ndarray
        1.0 (bullish), -1.0 (bearish), 0.0 (none).
    """
    # Convert Polars Series to numpy
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Ensure float64 and contiguous
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # TA‑Lib branch
    if use_talib and talib_available:
        talib_out = talib.CDL3LINESTRIKE(open_, high, low, close)
        # TA‑Lib returns 100, -100, 0 → convert to -1,0,1
        result = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(result, offset, fillna)
    # Numba branch with strict parameters
    out = _cdl_3linestrike_nb(
        open_, high, low, close, min_body_factor, max_shadow_factor, strict
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_3linestrike_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    strict: bool = False,
    min_body_factor: float = 0.0,
    max_shadow_factor: float = 1.0,
    output_col: str = "CDL_3LINESTRIKE",
) -> pl.DataFrame:
    """
    Add Three‑Line Strike column to Polars DataFrame.
    """
    out = cdl_3linestrike(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
        strict=strict,
        min_body_factor=min_body_factor,
        max_shadow_factor=max_shadow_factor,
    )
    return df.with_columns(pl.Series(output_col, out))