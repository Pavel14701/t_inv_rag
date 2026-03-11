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
def _cdl_highwave_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_body_factor: float,
    max_shadow_factor: float,
    strict: bool,
    symmetric: bool,  # API consistency
) -> np.ndarray:
    """
    Optimized High-Wave pattern.

    Returns:
        1.0 → bullish high-wave (symmetric mode)
       -1.0 → bearish high-wave (symmetric mode)
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        o0 = open_[i]
        c0 = close[i]
        h0 = high[i]
        l0 = low[i]
        rng = h0 - l0
        if rng <= 0.0:
            continue
        # Body
        body = c0 - o0 if c0 > o0 else o0 - c0
        # Shadows
        upper = h0 - (c0 if c0 > o0 else o0)
        lower = (c0 if c0 > o0 else o0) - l0
        # High-Wave shape:
        # - very small body
        # - very long upper & lower shadows
        if body > min_body_factor * rng:
            continue
        if upper < max_shadow_factor * rng:
            continue
        if lower < max_shadow_factor * rng:
            continue
        if strict:
            # Even stricter: shadows must dominate body
            if upper < 2.0 * body or lower < 2.0 * body:
                continue
        if symmetric:
            out[i] = 1.0 if c0 >= o0 else -1.0
        else:
            # TA-Lib returns +100 for both colors
            out[i] = 1.0
    return out


def cdl_highwave(
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
    max_shadow_factor: float = 0.3,
) -> np.ndarray:
    """
    High-Wave pattern with strict support.
    """
    if isinstance(open_, pl.Series): 
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series): 
        high = high.to_numpy()
    if isinstance(low, pl.Series): 
        low = low.to_numpy()
    if isinstance(close, pl.Series): 
        close = close.to_numpy()
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    # TA-Lib fallback
    if use_talib and talib_available and not symmetric:
        talib_out = talib.CDLHIGHWAVE(open_, high, low, close)
        result = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(result, offset, fillna)
    out = _cdl_highwave_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric,
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_highwave_polars(
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
    max_shadow_factor: float = 0.3,
    output_col: str = "CDL_HIGHWAVE",
) -> pl.DataFrame:
    out = cdl_highwave(
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
