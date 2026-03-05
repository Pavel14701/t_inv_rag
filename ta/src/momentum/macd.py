# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..overlap.ema import ema_ind
from ..utils import _apply_offset_fillna


def macd_numpy(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    asmode: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy‑based MACD calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    fast : int
        Fast EMA period.
    slow : int
        Slow EMA period.
    signal : int
        Signal EMA period.
    asmode : bool
        If True, use alternative AS mode.
    offset, fillna, use_talib : as usual.

    Returns
    -------
    tuple of np.ndarray
        (macd_line, signal_line, histogram)
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if use_talib and talib_available:
        macd, signalma, hist = talib.MACD(close, fast, slow, signal)
        if asmode:
            # AS mode: macd = macd - signalma, then recompute signal and hist
            macd = macd - signalma
            # need to recompute signal and hist from the new macd
            signalma = ema_ind(macd, length=signal, use_talib=False)
            hist = macd - signalma
    else:
        # Our own EMA-based MACD
        fast_ema = ema_ind(close, length=fast, use_talib=False)
        slow_ema = ema_ind(close, length=slow, use_talib=False)
        macd = fast_ema - slow_ema
        signalma = ema_ind(macd, length=signal, use_talib=False)
        hist = macd - signalma
        if asmode:
            macd = macd - signalma
            signalma = ema_ind(macd, length=signal, use_talib=False)
            hist = macd - signalma
    # Apply offset and fillna
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal MACD (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return macd_numpy(close, fast, slow, signal, asmode, offset, fillna, use_talib)


def macd_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    asmode: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add MACD columns to Polars DataFrame.

    Columns added:
        MACD{suffix}_{fast}_{slow}_{signal}
        MACD{suffix}s_{fast}_{slow}_{signal}
        MACD{suffix}h_{fast}_{slow}_{signal}

    If asmode=True, prefix becomes "MACDAS" instead of "MACD".

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    fast, slow, signal, asmode, offset, fillna, use_talib : as above.
    suffix : str
        Custom suffix (default f"_{fast}_{slow}_{signal}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new columns.
    """
    close = df[close_col].to_numpy()
    macd_line, signal_line, hist = macd_numpy(
        close, fast, slow, signal, asmode, offset, fillna, use_talib
    )
    suffix = suffix or f"_{fast}_{slow}_{signal}"
    prefix = "MACDAS" if asmode else "MACD"
    return pl.DataFrame({
        date_col: df[date_col],
        f"{prefix}{suffix}": macd_line,
        f"{prefix}s{suffix}": signal_line,
        f"{prefix}h{suffix}": hist,
    })