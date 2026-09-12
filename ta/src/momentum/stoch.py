# -*- coding: utf-8 -*-
"""Stochastic Oscillator (STOCH).

Full Stochastic: %K is the raw stochastic smoothed over ``smooth_k``
bars, %D is %K smoothed over ``d`` bars:

    raw = 100 * (close - LL) / (HH - LL)      # window k
    %K  = MA(raw,  smooth_k)
    %D  = MA(%K,   d)

Defaults: k=14, d=3, smooth_k=3, mamode='sma' (canonical values).
TA-Lib (STOCH) is used when available and ``mamode`` maps to a
TA-Lib MA type; otherwise ``ma_mode`` smoothing runs.

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- NaN propagates through the rolling extremes and the smoothing;
- a fully flat window (``HH == LL``) is 0/0-like: the raw value is
  NaN (undefined), never a fabricated value.
"""

from typing import cast

import numpy as np
import polars as pl

from .._array_ops import (
    _apply_offset_fillna,
    _rolling_max_numba,
    _rolling_min_numba,
)
from ..external import _TALIB_MA_MAP, talib, talib_available
from ..ma import ma_mode


def stoch_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the full Stochastic (%K, %D) using NumPy.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays (float64), same length.
    k : int
        Raw stochastic window (>= 1).
    d : int
        %D smoothing length (>= 1).
    smooth_k : int
        %K smoothing length (>= 1).
    mamode : str
        Smoothing MA mode (default 'sma').
    offset, fillna : as usual.
    use_talib : bool
        Prefer TA-Lib when ``mamode`` is supported by it.

    Returns
    -------
    tuple of np.ndarray
        (stoch_k, stoch_d) in [0, 100] (NaN on flat windows).

    Raises
    ------
    ValueError
        If ``k``, ``d`` or ``smooth_k`` < 1.

    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if d < 1:
        raise ValueError("d must be >= 1")
    if smooth_k < 1:
        raise ValueError("smooth_k must be >= 1")
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # Empty input is a valid degenerate case: return empty outputs rather
    # than raising inside the numba rolling kernels (TZ-14 stabilization).
    if high.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    # Numba rolling kernels require writable contiguous buffers.
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.writeable:
        low = low.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    ma_type = _TALIB_MA_MAP.get(mamode.lower())
    if use_talib and talib_available and ma_type is not None:
        k_arr, d_arr = talib.STOCH(
            high,
            low,
            close,
            fastk_period=k,
            slowk_period=smooth_k,
            slowk_matype=ma_type,
            slowd_period=d,
            slowd_matype=ma_type,
        )
        stoch_k = k_arr
        stoch_d = d_arr
    else:
        lowest_low = _rolling_min_numba(low, k)
        highest_high = _rolling_max_numba(high, k)
        denom = highest_high - lowest_low
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = 100.0 * (close - lowest_low) / denom
        raw = np.where(denom == 0.0, np.nan, raw)  # noqa: RUF069 - exact IEEE zero/sign check
        # Smoothing with nan_policy='ignore': the raw %K warm-up prefix
        # is inherently NaN and must not poison later windows.
        stoch_k = cast(
            np.ndarray,
            ma_mode(
                mamode,
                raw,
                length=smooth_k,
                offset=0,
                fillna=None,
                use_talib=False,
                nan_policy="ignore",
            ),
        )
        stoch_d = cast(
            np.ndarray,
            ma_mode(
                mamode,
                stoch_k,
                length=d,
                offset=0,
                fillna=None,
                use_talib=False,
                nan_policy="ignore",
            ),
        )
    stoch_k = _apply_offset_fillna(stoch_k, offset, fillna)
    stoch_d = _apply_offset_fillna(stoch_d, offset, fillna)
    return stoch_k, stoch_d


def stoch_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal Stochastic (accepts numpy array or Polars Series)."""
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return stoch_numpy(
        high,
        low,
        close,
        k,
        d,
        smooth_k,
        mamode,
        offset,
        fillna,
        use_talib,
    )


def stoch_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = "",
) -> pl.DataFrame:
    """Add Stochastic columns to a Polars DataFrame.

    Added columns: ``STOCHk{suffix}`` (%K) and ``STOCHd{suffix}``
    (%D) where suffix defaults to ``_{k}_{d}_{smooth_k}``.
    """
    high = df[high_col].cast(pl.Float64).to_numpy()
    low = df[low_col].cast(pl.Float64).to_numpy()
    close = df[close_col].cast(pl.Float64).to_numpy()
    stoch_k, stoch_d = stoch_numpy(
        high,
        low,
        close,
        k,
        d,
        smooth_k,
        mamode,
        offset,
        fillna,
        use_talib,
    )
    if not suffix:
        suffix = f"_{k}_{d}_{smooth_k}"
    return df.with_columns(
        [
            pl.Series(f"STOCHk{suffix}", stoch_k),
            pl.Series(f"STOCHd{suffix}", stoch_d),
        ]
    )
