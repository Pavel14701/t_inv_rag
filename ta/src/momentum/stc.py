# -*- coding: utf-8 -*-
"""Schaff Trend Cycle (STC).

A "cycle" version of MACD: two successive stochastic normalisations of
the MACD line, each smoothed by a short EMA:

    macd   = EMA(close, fast) - EMA(close, slow)
    stoch1 = EMA(stoch(macd,   tclen), factor)
    stc    = EMA(stoch(stoch1, tclen), factor)

Defaults follow the canonical definition (tclen=10, fast=12,
slow=26, factor=3). TA-Lib has no STC; the native path always runs.

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- NaN propagates through every stage;
- a flat normalisation window (max == min) yields NaN (0/0
  undefined), which then propagates through the EMA recursion;
- no fastmath, no fabricated values.
"""
import numpy as np
import polars as pl

from ..overlap.ema import ema_ind
from .._array_ops import (
    _apply_offset_fillna,
    _rolling_max_numba,
    _rolling_min_numba,
)


def _stoch_of_series(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling stochastic normalisation of a series:
    100 * (x - LL) / (HH - LL); flat windows -> NaN.
    """
    highest = _rolling_max_numba(x, window)
    lowest = _rolling_min_numba(x, window)
    denom = highest - lowest
    with np.errstate(divide='ignore', invalid='ignore'):
        out = 100.0 * (x - lowest) / denom
    return np.where(denom == 0.0, np.nan, out)


def _ema_of_warmup(x: np.ndarray, length: int) -> np.ndarray:
    """EMA of a series with a NaN warm-up prefix.

    Forward-fills the prefix with the first valid value (so the EMA
    recursion starts cleanly) and masks the contaminated prefix.
    """
    out = np.full(len(x), np.nan, dtype=np.float64)
    first_valid = np.argmax(~np.isnan(x))
    if np.isnan(x[first_valid]):
        return out  # all-NaN input stays all-NaN
    filled = x.copy()
    filled[:first_valid] = x[first_valid]
    out = ema_ind(
        filled, length=length, use_talib=False, nan_policy='ignore'
    )
    out[:first_valid + length - 1] = np.nan
    return out


def stc_numpy(
    close: np.ndarray,
    tclen: int = 10,
    fast: int = 12,
    slow: int = 26,
    factor: int = 3,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STC, the MACD line and the first smoothed stochastic.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    tclen : int
        Stochastic normalisation window (>= 1).
    fast, slow : int
        MACD EMA periods (>= 1).
    factor : int
        EMA smoothing applied to each stochastic stage (>= 1).
    offset, fillna : as usual.
    use_talib : bool
        Prefer TA-Lib for the inner EMAs when available.

    Returns
    -------
    tuple of np.ndarray
        (stc, macd, stoch) where ``stoch`` is the first smoothed
        stochastic of the MACD line.

    Raises
    ------
    ValueError
        If any period < 1.

    """
    if tclen < 1:
        raise ValueError('tclen must be >= 1')
    if fast < 1:
        raise ValueError('fast must be >= 1')
    if slow < 1:
        raise ValueError('slow must be >= 1')
    if factor < 1:
        raise ValueError('factor must be >= 1')
    close = np.asarray(close, dtype=np.float64, copy=False)
    if close.size == 0:
        return np.array([]), np.array([]), np.array([])
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    fast_ema = ema_ind(
        close, length=fast, use_talib=use_talib, nan_policy='ignore'
    )
    slow_ema = ema_ind(
        close, length=slow, use_talib=use_talib, nan_policy='ignore'
    )
    macd = fast_ema - slow_ema
    # Stage 1: stochastic of the MACD line + EMA smoothing.
    stoch = _ema_of_warmup(_stoch_of_series(macd, tclen), factor)
    # Stage 2: stochastic of stage 1 + EMA smoothing (the STC itself).
    stc = _ema_of_warmup(_stoch_of_series(stoch, tclen), factor)
    stc = _apply_offset_fillna(stc, offset, fillna)
    macd = _apply_offset_fillna(macd, offset, fillna)
    stoch = _apply_offset_fillna(stoch, offset, fillna)
    return stc, macd, stoch


def stc_ind(
    close: np.ndarray | pl.Series,
    tclen: int = 10,
    fast: int = 12,
    slow: int = 26,
    factor: int = 3,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Universal STC (accepts numpy array or Polars Series)."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return stc_numpy(
        close, tclen=tclen, fast=fast, slow=slow, factor=factor,
        offset=offset, fillna=fillna, use_talib=use_talib,
    )


def stc_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    tclen: int = 10,
    fast: int = 12,
    slow: int = 26,
    factor: int = 3,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = '',
) -> pl.DataFrame:
    """Add STC columns to a Polars DataFrame.

    Added columns: ``STC{suffix}``, ``STCmacd{suffix}`` and
    ``STCstoch{suffix}`` where suffix defaults to
    ``_{tclen}_{fast}_{slow}_{factor}``.
    """
    close = df[close_col].cast(pl.Float64).to_numpy()
    stc, macd, stoch = stc_numpy(
        close, tclen=tclen, fast=fast, slow=slow, factor=factor,
        offset=offset, fillna=fillna, use_talib=use_talib,
    )
    if not suffix:
        suffix = f'_{tclen}_{fast}_{slow}_{factor}'
    return df.with_columns([
        pl.Series(f'STC{suffix}', stc),
        pl.Series(f'STCmacd{suffix}', macd),
        pl.Series(f'STCstoch{suffix}', stoch),
    ])
