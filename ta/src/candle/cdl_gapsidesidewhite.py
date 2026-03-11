# TODO: Review
from numba import njit, types
import numpy as np
import polars as pl

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
def _cdl_gapsidesidewhite_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_body_factor: float,
    max_shadow_factor: float,
    strict: bool,
    symmetric: bool,  # для совместимости
) -> np.ndarray:
    """
    Optimized Gap Side-by-Side White Lines pattern.

    Returns:
        1.0 → bullish gap side-by-side white lines
       -1.0 → bearish gap side-by-side white lines
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(2, n):
        # Candle 1
        o2 = open_[i - 2]
        c2 = close[i - 2]
        h2 = high[i - 2]
        l2 = low[i - 2]

        # Candle 2
        o1 = open_[i - 1]
        c1 = close[i - 1]
        h1 = high[i - 1]
        l1 = low[i - 1]

        # Candle 3
        o0 = open_[i]
        c0 = close[i]
        h0 = high[i]
        l0 = low[i]

        r2 = h2 - l2
        r1 = h1 - l1
        r0 = h0 - l0
        if r2 <= 0.0 or r1 <= 0.0 or r0 <= 0.0:
            continue

        # Bodies
        b2 = c2 - o2 if c2 > o2 else o2 - c2
        b1 = c1 - o1 if c1 > o1 else o1 - c1
        b0 = c0 - o0 if c0 > o0 else o0 - c0
        if b2 <= 0.0 or b1 <= 0.0 or b0 <= 0.0:
            continue

        # Colors
        bull2 = c2 > o2
        bull1 = c1 > o1
        bull0 = c0 > o0
        bear2 = c2 < o2
        bear1 = c1 < o1
        bear0 = c0 < o0

        direction = 0.0

        # Bullish gap side-by-side white lines:
        # 1) first candle bullish
        # 2) second & third bullish
        # 3) gap up between 1 и 2
        # 4) 2 и 3 примерно на одном уровне (close/ open близки)
        if bull2 and bull1 and bull0:
            if l1 > h2:  # gap up
                # 2 и 3 примерно на одном уровне
                if abs(o0 - o1) <= 0.25 * (r1 + r0) and abs(c0 - c1) <= 0.25 * (r1 + r0):
                    direction = 1.0

        # Bearish gap side-by-side white lines:
        # 1) first candle bearish
        # 2) second & third bullish
        # 3) gap down between 1 и 2
        # 4) 2 и 3 примерно на одном уровне
        if direction == 0.0 and bear2 and bull1 and bull0:
            if h1 < l2:  # gap down
                if abs(o0 - o1) <= 0.25 * (r1 + r0) and abs(c0 - c1) <= 0.25 * (r1 + r0):
                    direction = -1.0

        if direction == 0.0:
            continue

        if strict:
            # Минимальный размер тел
            if (b2 < min_body_factor * r2 or
                b1 < min_body_factor * r1 or
                b0 < min_body_factor * r0):
                continue

            # Тени (быстро, без max/min)
            up2 = c2 if bull2 else o2
            lo2 = o2 if bull2 else c2
            sh2 = (h2 - up2) + (lo2 - l2)

            up1 = c1
            lo1 = o1
            sh1 = (h1 - up1) + (lo1 - l1)

            up0 = c0
            lo0 = o0
            sh0 = (h0 - up0) + (lo0 - l0)

            if (sh2 > max_shadow_factor * r2 or
                sh1 > max_shadow_factor * r1 or
                sh0 > max_shadow_factor * r0):
                continue

        out[i] = direction

    return out


def cdl_gapsidesidewhite(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    strict: bool = False,
    symmetric: bool = False,
    min_body_factor: float = 0.3,
    max_shadow_factor: float = 0.5,
) -> np.ndarray:
    """
    Gap Side-by-Side White Lines pattern with strict support.
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
        talib_out = talib.CDLGAPSIDESIDEWHITE(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    out = _cdl_gapsidesidewhite_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric,
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_gapsidesidewhite_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    strict: bool = False,
    symmetric: bool = False,
    min_body_factor: float = 0.3,
    max_shadow_factor: float = 0.5,
    output_col: str = "CDL_GAPSIDESIDEWHITE",
) -> pl.DataFrame:
    out = cdl_gapsidesidewhite(
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
