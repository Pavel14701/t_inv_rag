# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, jit

from ..utils import _apply_offset_fillna
from . import rsi_ind


# ----------------------------------------------------------------------
# Streak calculation (Numba) – corrected version
# ----------------------------------------------------------------------
@jit((float64[:],), nopython=True, fastmath=True, cache=True)
def _streak_numba(close: np.ndarray) -> np.ndarray:
    """
    Calculate streak: length of consecutive up/down moves.
    Returns array of cumulative counts (positive for up, negative for down).
    """
    n = len(close)
    streak = np.zeros(n, dtype=np.float64)
    if n < 2:
        return streak
    for i in range(1, n):
        if close[i] > close[i - 1]:
            # Up move
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif close[i] < close[i - 1]:
            # Down move
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
        else:
            # No change resets streak
            streak[i] = 0
    return streak


# ----------------------------------------------------------------------
# Percent Rank (rolling) – Numba, strict comparison, direct indexing
# ----------------------------------------------------------------------
@jit((float64[:], int64), nopython=True, fastmath=True, cache=True)
def _percent_rank_numba(close: np.ndarray, length: int) -> np.ndarray:
    """
    Rolling Percent Rank: for each window, compute percentage of values
    strictly less than the current value.
    Uses direct indexing (no temporary window slices).
    Result in range 0..100.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length or length < 2:
        # Percent rank requires at least 2 periods to compute meaningful percentage
        return out
    for i in range(length - 1, n):
        current = close[i]
        count_lt = 0
        # Loop over window indices directly
        for j in range(i - length + 1, i + 1):
            if close[j] < current:
                count_lt += 1
        # Formula: (count of strictly less) / (length - 1) * 100
        out[i] = count_lt / (length - 1) * 100.0
    return out


# ----------------------------------------------------------------------
# Main CRSI calculation with NaN handling, validation and normalization
# ----------------------------------------------------------------------
def crsi_numpy(
    close: np.ndarray,
    rsi_length: int = 3,
    streak_length: int = 2,
    rank_length: int = 100,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',          # 'raise', 'ffill', 'bfill', 'both'
    normalize: bool = False,             # if True, replace NaN in result with 50.0
) -> np.ndarray:
    """
    Numpy‑based Connors RSI calculation with NaN handling.

    Parameters
    ----------
    close : np.ndarray
        Array of close prices.
    rsi_length, streak_length, rank_length : int
        Periods for RSI components. rsi_length and streak_length must be >= 1,
        rank_length must be >= 2.
    scalar : float
        Scaling factor (typically 100).
    offset, fillna : as usual.
    use_talib : bool
        Whether to use TA-Lib for RSI (if available).
    nan_policy : str
        How to handle NaN values in input:
        - 'raise': raise ValueError if any NaN found.
        - 'ffill': forward fill (propagate last valid observation).
        - 'bfill': backward fill (propagate next valid observation).
        - 'both': first forward fill, then backward fill (fills all gaps).
    normalize : bool
        If True, replace any remaining NaN in the final CRSI with 50.0
        (neutral value). Useful for machine learning pipelines.

    Returns
    -------
    np.ndarray
        CRSI values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    # ---- Input validation ----
    if rsi_length < 1:
        raise ValueError("rsi_length must be >= 1")
    if streak_length < 1:
        raise ValueError("streak_length must be >= 1")
    if rank_length < 2:
        raise ValueError("rank_length must be >= 2 for percent rank")
    # ---- NaN handling on input ----
    if np.isnan(close).any():
        if nan_policy == 'raise':
            raise ValueError("Input contains NaN values. \
                Use nan_policy='ffill', 'bfill' or 'both' to fill them.")
        elif nan_policy == 'ffill':
            # Forward fill
            close = close.copy()
            for i in range(1, len(close)):
                if np.isnan(close[i]):
                    close[i] = close[i - 1]
        elif nan_policy == 'bfill':
            # Backward fill
            close = close.copy()
            for i in range(len(close) - 2, -1, -1):
                if np.isnan(close[i]):
                    close[i] = close[i + 1]
        elif nan_policy == 'both':
            # First forward fill, then backward fill (fills all gaps)
            close = close.copy()
            # forward fill
            for i in range(1, len(close)):
                if np.isnan(close[i]):
                    close[i] = close[i - 1]
            # backward fill (to handle leading NaNs)
            for i in range(len(close) - 2, -1, -1):
                if np.isnan(close[i]):
                    close[i] = close[i + 1]
        else:
            raise ValueError(f"Unknown nan_policy: {nan_policy}. \
                Use 'raise', 'ffill', 'bfill', or 'both'.")
    # Ensure C-contiguous for Numba performance
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # 1. RSI of price
    rsi_price = rsi_ind(close, length=rsi_length, scalar=scalar, use_talib=use_talib)
    # 2. Streak and its RSI
    streak = _streak_numba(close)
    rsi_streak = rsi_ind(
        streak, length=streak_length, scalar=scalar, use_talib=use_talib
    )
    # 3. Percent Rank of price
    percent_rank = _percent_rank_numba(close, rank_length)
    # 4. Average
    crsi = (rsi_price + rsi_streak + percent_rank) / 3.0
    # ---- Normalize (replace NaN with 50.0) if requested ----
    if normalize:
        crsi = np.where(np.isnan(crsi), 50.0, crsi)
    # Apply offset and fillna (fillna only affects remaining NaN if normalize=False)
    return _apply_offset_fillna(crsi, offset, fillna)


def crsi_ind(
    close: np.ndarray | pl.Series,
    rsi_length: int = 3,
    streak_length: int = 2,
    rank_length: int = 100,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    normalize: bool = False,
) -> np.ndarray:
    """
    Universal Connors RSI (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return crsi_numpy(close, rsi_length, streak_length, rank_length,
                      scalar, offset, fillna, use_talib, nan_policy, normalize)


def crsi_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    rsi_length: int = 3,
    streak_length: int = 2,
    rank_length: int = 100,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    normalize: bool = False,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add CRSI column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    rsi_length, streak_length, rank_length, scalar, \
        offset, fillna, use_talib, nan_policy, normalize : as above.
    output_col : str, optional
        Output column name (default f"CRSI_{rsi_length}_{streak_length}_{rank_length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.
    """
    close = df[close_col].to_numpy()
    result = crsi_numpy(close, rsi_length, streak_length, rank_length,
                        scalar, offset, fillna, use_talib, nan_policy, normalize)
    out_name = output_col or f"CRSI_{rsi_length}_{streak_length}_{rank_length}"
    return df.with_columns([pl.Series(out_name, result)])