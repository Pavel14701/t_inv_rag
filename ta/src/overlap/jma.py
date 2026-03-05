from typing import Optional

import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Numba‑compiled core of the Jurik Moving Average (most aggressive version)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _jma_numba_core(
    close: np.ndarray,
    length_param: int,
    phase: float
) -> np.ndarray:
    """
    Jurik Moving Average core loop – most aggressive Numba implementation.

    This function performs the entire JMA calculation in a single pass,
    using a circular buffer for the rolling volatility sum to avoid an
    inner loop. All constants are pre‑computed.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of close prices.
    length_param : int
        Main period of the JMA (typically 7).
    phase : float
        Phase parameter in the range [-100, 100]; controls the lag/smoothness trade‑off.

    Returns
    -------
    np.ndarray
        JMA values. The first `length_param - 1` elements are set to NaN,
        matching the original pandas_ta behaviour.
    """
    n = len(close)
    # Output array and two helper arrays
    jma = np.empty(n, dtype=np.float64)
    volty = np.empty(n, dtype=np.float64)
    v_sum = np.empty(n, dtype=np.float64)
    sum_length = 10  # window for volatility sum (fixed in original)
    L = 0.5 * (length_param - 1)  # transformed length
    # Phase factor
    if phase < -100.0:
        pr = 0.5
    elif phase > 100.0:
        pr = 2.5
    else:
        pr = 1.5 + phase * 0.01
    length1 = max(np.log(np.sqrt(L)) / np.log(2.0) + 2.0, 0.0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * np.sqrt(L)
    bet = length2 / (length2 + 1.0)
    beta = 0.45 * (length_param - 1) / (0.45 * (length_param - 1) + 2.0)
    limit = length1 ** (1.0 / pow1)  # pre‑computed upper bound for d_volty
    jma[0] = close[0]
    volty[0] = 0.0
    v_sum[0] = 0.0
    ma1 = close[0]
    uBand = close[0]
    lBand = close[0]
    det0 = 0.0
    det1 = 0.0
    # We need a 66‑period average of v_sum. Instead of recalculating
    # the sum from scratch each time, we maintain a running total.
    window_size = 66
    v_sum_window = np.zeros(window_size, dtype=np.float64)
    v_sum_idx = 0
    v_sum_total = 0.0
    window_filled = False
    # Main loop
    for i in range(1, n):
        price = close[i]
        # Price volatility
        del1 = price - uBand
        del2 = price - lBand
        if abs(del1) != abs(del2):
            volty[i] = max(abs(del1), abs(del2))
        else:
            volty[i] = 0.0
        # Running sum of volatility over `sum_length` periods
        past_idx = i - sum_length
        if past_idx < 0:
            past_idx = 0
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[past_idx]) / sum_length
        # Update running window sum for v_sum (rolling average of last 66)
        if i < window_size:
            v_sum_total += v_sum[i]
            v_sum_window[i] = v_sum[i]
        else:
            oldest = v_sum_window[v_sum_idx]
            v_sum_total = v_sum_total - oldest + v_sum[i]
            v_sum_window[v_sum_idx] = v_sum[i]
            v_sum_idx = (v_sum_idx + 1) % window_size
            window_filled = True
        # Average volatility over the last 66 values of v_sum
        if window_filled:
            avg_volty = v_sum_total / window_size
        else:
            avg_volty = v_sum_total / (i + 1) if i > 0 else 0.0
        #  Relative volatility factor
        d_volty = 0.0 if avg_volty == 0.0 else volty[i] / avg_volty
        if d_volty < 1.0:
            r_volty = 1.0
        elif d_volty > limit:
            r_volty = limit
        else:
            r_volty = d_volty
        # Jurik volatility bands
        power = r_volty ** pow1
        kv = bet ** np.sqrt(power)
        if del1 > 0.0:
            uBand = price
        else:
            uBand = price - kv * del1
        if del2 < 0.0:
            lBand = price
        else:
            lBand = price - kv * del2
        # Jurik dynamic factor
        alpha = beta ** power
        # 1st stage – adaptive EMA
        ma1 = (1.0 - alpha) * price + alpha * ma1
        # 2nd stage – Kalman‑like filter
        det0 = (1.0 - beta) * (price - ma1) + beta * det0
        ma2 = ma1 + pr * det0
        # 3rd stage – final smoothing
        det1 = (
            (ma2 - jma[i - 1]) * (1.0 - alpha) * (1.0 - alpha)
        ) + (alpha * alpha * det1)
        jma[i] = jma[i - 1] + det1
    # Set first `length_param - 1` values to NaN (original behaviour)
    for i in range(length_param - 1):
        jma[i] = np.nan
    return jma


# ----------------------------------------------------------------------
# Public functions (Numba + Polars integration)
# ----------------------------------------------------------------------
def jma_numba(
    close: np.ndarray,
    length: int = 7,
    phase: float = 0.0,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Jurik Moving Average using the 
    aggressively optimized Numba core (raw numpy version).

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Main period of the JMA (default 7).
    phase : float
        Phase parameter in the range [-100, 100] (default 0.0).
    offset : int
        Shift the result (positive = forward, negative = backward).
    fillna : float, optional
        Value to replace any remaining NaNs after shifting.

    Returns
    -------
    np.ndarray
        JMA values with the same length as `close`. The first `length-1`
        values are NaN.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    jma = _jma_numba_core(close, length, phase)
    return _apply_offset_fillna(jma, offset, fillna)


def jma_ind(
    close: np.ndarray | pl.Series,
    length: int = 7,
    phase: float = 0.0,
    offset: int = 0,
    fillna: Optional[float] = None
) -> np.ndarray:
    """
    Universal Jurik Moving Average (accepts numpy array or Polars Series).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Input price series.
    length : int
        JMA period (default 7).
    phase : float
        Phase parameter [-100, 100] (default 0.0).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        JMA values as a numpy array.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return jma_numba(close, length, phase, offset, fillna)


def jma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 7,
    phase: float = 0.0,
    offset: int = 0,
    fillna: Optional[float] = None,
    output_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Jurik Moving Average for Polars DataFrames.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column containing close prices.
    length : int
        JMA period (default 7).
    phase : float
        Phase parameter [-100, 100] (default 0.0).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Name of the output column. If not provided,
        it defaults to f"JMA_{length}_{phase}".

    Returns
    -------
    pl.DataFrame
        Original polars DataFrame with the JMA values.
    """
    close = df[close_col].to_numpy()
    result = jma_ind(close, length, phase, offset, fillna)
    out_name = output_col or f"JMA_{length}_{phase}"
    return df.with_columns([pl.Series(out_name, result)])