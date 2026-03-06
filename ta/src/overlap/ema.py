# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from .. import talib, talib_available
from ..utils import _apply_offset_fillna, _handle_nan_policy


# ----------------------------------------------------------------------
# Optimized EMA using Numba (nopython mode, fastmath)
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True, parallel=False)
def _ema_numba_opt(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Exponential Moving Average (optimized Numba version).

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array of prices (assumed to have no NaNs).
    window : int
        EMA period.

    Returns
    -------
    np.ndarray
        EMA array with same length as input; first (window-1) values are NaN.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    alpha = 2.0 / (window + 1)
    s = 0.0
    for i in range(window):
        s += arr[i]
    out[window - 1] = s / window
    for i in range(window, n):
        out[i] = out[i - 1] + alpha * (arr[i] - out[i - 1])
    return out


# ----------------------------------------------------------------------
# EMA using Numba (with NaN handling, trim, etc.)
# ----------------------------------------------------------------------
def ema_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """
    Exponential Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        EMA period.
    offset, fillna, nan_policy, trim : as usual.
    """
    # ---- Input validation ----
    if length < 1:
        raise ValueError("EMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64)

    # Check for infinite values
    if np.isinf(close).any():
        raise ValueError("Input contains non-finite values (inf or -inf).")

    # Apply NaN policy
    close = _handle_nan_policy(close, nan_policy, "close")

    # Ensure C-contiguous
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    # Check length against required window
    if len(close) < length:
        raise ValueError(
            f"Input series too short: need at least \
                {length} elements, got {len(close)}."
        )
    # Calculate EMA
    ema = _ema_numba_opt(close, length)
    # Trim if requested
    if trim:
        valid_start = length - 1
        if valid_start < len(ema):
            ema = ema[valid_start:]
        else:
            ema = np.array([])
    # Apply offset and fillna
    return _apply_offset_fillna(ema, offset, fillna)


# ----------------------------------------------------------------------
# EMA using TA-Lib (with NaN handling – TA‑Lib itself doesn't handle NaNs)
# ----------------------------------------------------------------------
def ema_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """EMA via TA-Lib, with pre‑processing of NaNs."""
    if not talib_available:
        raise ImportError("TA-Lib not available")
    if length < 1:
        raise ValueError("EMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64)
    if np.isinf(close).any():
        raise ValueError("Input contains non-finite values (inf or -inf).")
    # TA‑Lib does not handle NaNs, so we pre‑process
    close = _handle_nan_policy(close, nan_policy, "close")
    if len(close) < length:
        raise ValueError(
            f"Input series too short: need at least \
                {length} elements, got {len(close)}."
        )
    ema = talib.EMA(close, timeperiod=length)
    if trim:
        valid_start = length - 1
        if valid_start < len(ema):
            ema = ema[valid_start:]
        else:
            ema = np.array([])
    return _apply_offset_fillna(ema, offset, fillna)


# ----------------------------------------------------------------------
# Universal EMA function
# ----------------------------------------------------------------------
def ema_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Universal EMA with automatic backend selection."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = close.astype(np.float64)

    if use_talib and talib_available:
        return ema_talib(
            close,
            length=length,
            offset=offset,
            fillna=fillna,
            nan_policy=nan_policy,
            trim=trim,
        )
    else:
        return ema_numba(
            close,
            length=length,
            offset=offset,
            fillna=fillna,
            nan_policy=nan_policy,
            trim=trim,
        )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def ema_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None,
) -> pl.DataFrame:
    """EMA for Polars DataFrame (returns same length, no trim)."""
    close = df[close_col].to_numpy()
    result = ema_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,          # Polars всегда возвращает полную длину
    )
    out_name = output_col or f"EMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])