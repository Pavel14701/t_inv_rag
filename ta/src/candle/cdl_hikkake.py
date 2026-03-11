# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import njit, types

from .. import talib, talib_available
from ..utils import _apply_offset_fillna


@njit(
    (
        types.float64[:], types.float64[:], types.float64[:], types.float64[:],
        types.int64, types.boolean
    ),
    nopython=True,
    cache=True,
    fastmath=True,
)
def _cdl_hikkake_nb(
    open_, high, low, close,
    lookahead,
    strict,
):
    """
    Optimized Hikkake pattern (approximation of TA-Lib logic).

    Returns:
        1.0 → bullish hikkake (bear trap)
       -1.0 → bearish hikkake (bull trap)
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, np.float64)
    for i in range(2, n - lookahead):
        # Bar -2, -1, 0 (inside bar setup)
        h2 = high[i - 2]
        l2 = low[i - 2]
        h1 = high[i - 1]
        l1 = low[i - 1]
        h0 = high[i]
        l0 = low[i]
        # Inside bar: bar -1 inside bar -2
        if not (h1 < h2 and l1 > l2):
            continue
        # Bar 0: breakout of inside bar range
        bullish_break = h0 > h1 and l0 >= l1
        bearish_break = l0 < l1 and h0 <= h1
        if not (bullish_break or bearish_break):
            continue
        direction = 0.0
        # Look for failure (reversal) in next `lookahead` bars
        if bullish_break:
            # Initial breakout up → watch for move below inside bar low
            for k in range(1, lookahead + 1):
                if low[i + k] < l1:
                    direction = -1.0  # bearish hikkake (bull trap)
                    break
        elif bearish_break:
            # Initial breakout down → watch for move above inside bar high
            for k in range(1, lookahead + 1):
                if high[i + k] > h1:
                    direction = 1.0  # bullish hikkake (bear trap)
                    break
        if direction == 0.0:
            continue
        if strict:
            # Простое доп. условие: диапазон inside bar не слишком мал
            rng2 = h2 - l2
            rng1 = h1 - l1
            if rng2 <= 0.0 or rng1 <= 0.0:
                continue
            if rng1 < 0.2 * rng2:
                continue
        # Сигнал ставим на баре подтверждения (i + k), но для простоты — на баре 0
        out[i] = direction
    return out


def cdl_hikkake(
    open_, high, low, close,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    strict: bool = False,
    lookahead: int = 3,
) -> np.ndarray:
    """
    Hikkake pattern with strict support and optional TA-Lib fallback.

    Returns:
        1.0 → bullish hikkake
       -1.0 → bearish hikkake
        0.0 → none
    """
    # Polars → numpy
    if isinstance(open_, pl.Series): 
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series): 
        high = high.to_numpy()
    if isinstance(low, pl.Series): 
        low = low.to_numpy()
    if isinstance(close, pl.Series): 
        close = close.to_numpy()
    open_ = np.asarray(open_, np.float64)
    high = np.asarray(high, np.float64)
    low = np.asarray(low, np.float64)
    close = np.asarray(close, np.float64)
    # TA-Lib fallback (если есть)
    if use_talib and talib_available:
        out = talib.CDLHIKKAKE(open_, high, low, close).astype(np.float64) / 100.0
        return _apply_offset_fillna(out, offset, fillna)
    out = _cdl_hikkake_nb(open_, high, low, close, lookahead, strict)
    return _apply_offset_fillna(out, offset, fillna)


def cdl_hikkake_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    strict: bool = False,
    lookahead: int = 3,
    output_col: str = "CDL_HIKKAKE",
) -> pl.DataFrame:
    out = cdl_hikkake(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
        strict=strict,
        lookahead=lookahead,
    )
    return df.with_columns(pl.Series(output_col, out))
