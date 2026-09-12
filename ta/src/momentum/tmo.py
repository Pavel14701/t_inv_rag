# -*- coding: utf-8 -*-
"""True Momentum Oscillator (TMO).

Rolling sum of per-bar "true momentum" with an EMA signal line:

    mom[i]  = open[i] - close[i - drift]
    main    = sum(mom, length)
    signal  = EMA(main, smooth)
    normalize: both scaled by 100 / length

Positive main = bullish pressure, negative = bearish. TA-Lib has no
TMO; the native path always runs.

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- NaN propagates: a window touching a NaN bar yields NaN main;
- the first ``length + drift - 1`` bars are NaN (warm-up).
"""
import numpy as np
import polars as pl
from numba import float64, int64, njit

from ..overlap.ema import ema_ind
from .._array_ops import _apply_offset_fillna


@njit(
    (float64[:], float64[:], int64, int64),
    fastmath=False,
    cache=True,
)
def _tmo_main_numba(
    open_: np.ndarray,
    close: np.ndarray,
    length: int,
    drift: int,
) -> np.ndarray:
    """Rolling sum of (open[i] - close[i - drift]) over ``length``
    bars; any NaN in a window poisons that window only.
    """
    n = len(open_)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(length + drift - 1, n):
        acc = 0.0
        bad = False
        for j in range(i - length + 1, i + 1):
            v = open_[j] - close[j - drift]
            if np.isnan(v):
                bad = True
                break
            acc += v
        if not bad:
            out[i] = acc
    return out


def tmo_numpy(
    open_: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    drift: int = 1,
    smooth: int = 4,
    normalize: bool = True,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute TMO main line and signal using NumPy.

    Parameters
    ----------
    open_, close : np.ndarray
        Price arrays (float64), same length.
    length : int
        Rolling-sum window (>= 1). Warm-up: first
        ``length + drift - 1`` bars are NaN.
    drift : int
        Momentum lookback (>= 1).
    smooth : int
        EMA length of the signal line (>= 1).
    normalize : bool
        If True, both lines are scaled by ``100 / length``.
    offset, fillna : as usual.

    Returns
    -------
    tuple of np.ndarray
        (main, signal).

    Raises
    ------
    ValueError
        If ``length``, ``drift`` or ``smooth`` < 1.

    """
    if length < 1:
        raise ValueError('length must be >= 1')
    if drift < 1:
        raise ValueError('drift must be >= 1')
    if smooth < 1:
        raise ValueError('smooth must be >= 1')
    open_ = np.asarray(open_, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    if open_.size == 0:
        return np.array([]), np.array([])
    if not open_.flags.writeable:
        open_ = open_.copy()
    if not open_.flags.c_contiguous:
        open_ = np.ascontiguousarray(open_)
    if not close.flags.writeable:
        close = close.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    main = _tmo_main_numba(open_, close, length, drift)
    # Signal line: forward-fill the warm-up NaN prefix with the first
    # valid value so the EMA recursion starts cleanly, then mask the
    # standard prefix (same approach as macd_numpy).
    main_filled = main.copy()
    first_valid = np.argmax(~np.isnan(main))
    if not np.isnan(main[first_valid]):
        main_filled[:first_valid] = main[first_valid]
    signalma = ema_ind(
        main_filled, length=smooth, use_talib=False, nan_policy='ignore'
    )
    signalma[:length + drift + smooth - 2] = np.nan
    if normalize:
        main = main * 100.0 / length
        signalma = signalma * 100.0 / length
    main = _apply_offset_fillna(main, offset, fillna)
    signalma = _apply_offset_fillna(signalma, offset, fillna)
    return main, signalma


def tmo_ind(
    open_: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    drift: int = 1,
    smooth: int = 4,
    normalize: bool = True,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal TMO (accepts numpy array or Polars Series)."""
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return tmo_numpy(
        open_, close, length=length, drift=drift, smooth=smooth,
        normalize=normalize, offset=offset, fillna=fillna,
    )


def tmo_polars(
    df: pl.DataFrame,
    open_col: str = 'open',
    close_col: str = 'close',
    length: int = 14,
    drift: int = 1,
    smooth: int = 4,
    normalize: bool = True,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = '',
) -> pl.DataFrame:
    """Add TMO columns to a Polars DataFrame.

    Added columns: ``TMO{suffix}`` (main) and ``TMOS{suffix}``
    (signal) where suffix defaults to ``_{length}``.
    """
    open_ = df[open_col].cast(pl.Float64).to_numpy()
    close = df[close_col].cast(pl.Float64).to_numpy()
    main, signalma = tmo_numpy(
        open_, close, length=length, drift=drift, smooth=smooth,
        normalize=normalize, offset=offset, fillna=fillna,
    )
    if not suffix:
        suffix = f'_{length}'
    return df.with_columns([
        pl.Series(f'TMO{suffix}', main),
        pl.Series(f'TMOS{suffix}', signalma),
    ])
