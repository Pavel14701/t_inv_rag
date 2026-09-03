# -*- coding: utf-8 -*-
"""MACD (Moving Average Convergence Divergence) indicator implementation.
Supports NumPy arrays, Polars DataFrames, and optional TA-Lib acceleration.
"""
import numpy as np
import polars as pl

from ..external import talib, talib_available
from ..overlap.ema import ema_ind
from .._array_ops import _apply_offset_fillna, replace_inf_with_nan


def macd_numpy(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    asmode: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'ignore',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MACD using NumPy.

    Parameters
    ----------
    close : np.ndarray
        Array of closing prices (float64).
    fast : int, default 12
        Fast EMA period.
    slow : int, default 26
        Slow EMA period.
    signal : int, default 9
        Signal EMA period.
    asmode : bool, default False
        If True, the MACD line is defined as
        (fast_ema - slow_ema) - signal_line, and then
        signal and histogram are recomputed from this new line.
    offset : int, default 0
        Number of initial periods to set to `fillna`.
    fillna : float or None, default None
        Value to fill the first `offset` elements.
    use_talib : bool, default True
        Attempt to use TA-Lib for calculation if available.
    nan_policy : str, default 'ignore'
        How to handle NaN values in the input. Passed to `ema_ind`.
        Options: 'raise', 'ignore', 'ffill', 'bfill', 'both'.

    Returns
    -------
    tuple of np.ndarray
        (macd_line, signal_line, histogram) - all arrays have the same length
        as `close`.

    Notes
    -----
    All floating-point operations follow IEEE 754 rules. `NaN` and `inf`
    are propagated naturally; no exceptions are raised.

    For the signal line, we internally forward-fill the MACD line to obtain
    a continuous series for the EMA calculation, then mask the initial
    (slow + signal - 2) values to NaN. This ensures correct alignment with
    the standard MACD definition.

    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if close.size == 0:
        return np.array([]), np.array([]), np.array([])
    if close.size < min(fast, slow):
        return (
            np.full(close.size, np.nan),
            np.full(close.size, np.nan),
            np.full(close.size, np.nan),
        )
    close = close.copy()
    replace_inf_with_nan(close)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # ---- Compute MACD line (fast_ema - slow_ema) ----
    if use_talib and talib_available:
        # TA-Lib returns MACD, signal, hist; we only use MACD
        macd, _, _ = talib.MACD(close, fast, slow, signal)
    else:
        fast_ema = ema_ind(
            close,
            length=fast,
            use_talib=False,
            nan_policy=nan_policy
        )
        slow_ema = ema_ind(
            close,
            length=slow,
            use_talib=False,
            nan_policy=nan_policy
        )
        macd = fast_ema - slow_ema
    # ---- Compute signal line and histogram using our logic ----
    # 1. Forward-fill leading NaNs in macd with the first valid value
    macd_filled = macd.copy()
    first_valid = np.argmax(~np.isnan(macd))
    if not np.isnan(macd[first_valid]):
        macd_filled[:first_valid] = macd[first_valid]
    # 2. Compute EMA of the filled macd (signal line)
    signalma = ema_ind(
        macd_filled,
        length=signal,
        use_talib=False,
        nan_policy='ignore'
    )
    # 3. Mask initial values to NaN: first (slow + signal - 2) elements
    signalma[:slow + signal - 2] = np.nan
    # 4. Histogram = macd - signalma
    hist = macd - signalma
    # ---- AS mode (if requested) ----
    if asmode:
        macd = macd - signalma
        # Recompute signal and hist from new macd
        macd_filled = macd.copy()
        first_valid = np.argmax(~np.isnan(macd))
        if not np.isnan(macd[first_valid]):
            macd_filled[:first_valid] = macd[first_valid]
        signalma = ema_ind(
            macd_filled,
            length=signal,
            use_talib=False,
            nan_policy='ignore'
        )
        signalma[:slow + signal - 2] = np.nan
        hist = macd - signalma
    # ---- Apply offset and fillna ----
    macd = _apply_offset_fillna(macd, offset, fillna)
    signalma = _apply_offset_fillna(signalma, offset, fillna)
    hist = _apply_offset_fillna(hist, offset, fillna)
    return macd, signalma, hist


def macd_ind(
    close: np.ndarray | pl.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    asmode: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'ignore',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Universal wrapper for MACD that accepts either a NumPy
    array or a Polars Series.

    All parameters are the same as in `macd_numpy`.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return macd_numpy(
        close, fast, slow, signal,
        asmode, offset, fillna, use_talib, nan_policy
    )


def macd_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    date_col: str = 'date',
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    asmode: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = '',
    nan_policy: str = 'ignore',
) -> pl.DataFrame:
    """Add MACD columns to a Polars DataFrame.

    The function **adds** new columns to the original DataFrame
    (does not replace it).

    Added columns (names depend on `asmode` and `suffix`):
        - If `asmode=False`:
            MACD{suffix}_{fast}_{slow}_{signal}
            MACD{suffix}s_{fast}_{slow}_{signal}
            MACD{suffix}h_{fast}_{slow}_{signal}
        - If `asmode=True`:
            MACDAS{suffix}_{fast}_{slow}_{signal}
            MACDAS{suffix}s_{fast}_{slow}_{signal}
            MACDAS{suffix}h_{fast}_{slow}_{signal}

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing close prices.
    date_col : str, default 'date'
        Name of the date column (used only to preserve order; not modified).
    fast, slow, signal, asmode, offset, fillna, use_talib :
        Same as in `macd_numpy`.
    suffix : str, default ''
        Custom suffix for column names. If empty, a default suffix
        f'_{fast}_{slow}_{signal}' is used.
    nan_policy : str, default 'ignore'
        Passed to `macd_numpy`.

    Returns
    -------
    pl.DataFrame
        The original DataFrame with the three new MACD columns appended.

    """
    close = df[close_col].to_numpy()
    macd_line, signal_line, hist = macd_numpy(
        close, fast, slow, signal,
        asmode, offset, fillna, use_talib, nan_policy
    )
    if not suffix:
        suffix = f'_{fast}_{slow}_{signal}'
    prefix = 'MACDAS' if asmode else 'MACD'
    return df.with_columns([
        pl.Series(f'{prefix}{suffix}', macd_line),
        pl.Series(f'{prefix}s{suffix}', signal_line),
        pl.Series(f'{prefix}h{suffix}', hist),
    ])
