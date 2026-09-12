# -*- coding: utf-8 -*-
"""Percentage Price Oscillator (PPO).

Percentage version of MACD:

    PPO  = scalar * (EMA(fast) - EMA(slow)) / EMA(slow)
    PPOs = EMA(PPO, signal)
   PPOh  = PPO - PPOs

TA-Lib is used for the PPO line when available and ``scalar == 100``
(TA-Lib semantics match exactly); the signal line and histogram are
always computed natively, exactly as in ``macd_numpy``.

IEEE 754 notes
--------------
- ``+/-Inf`` on input is converted to ``NaN`` (no input mutation);
- NaN propagates through the EMAs and the division;
- a zero slow EMA yields ``x/0`` -> +/-Inf (kept, documented) and
  ``0/0`` -> NaN.
"""
import numpy as np
import polars as pl

from ..external import talib, talib_available
from ..overlap.ema import ema_ind
from .._array_ops import _apply_offset_fillna, replace_inf_with_nan


def ppo_numpy(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'ignore',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PPO using NumPy.

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
    scalar : float, default 100.0
        Scaling factor (100 -> percentage oscillator).
    offset : int, default 0
        Shift of the output series.
    fillna : float, optional
        Replacement for warm-up/shifted-in NaNs.
    use_talib : bool, default True
        Prefer TA-Lib for the PPO line when ``scalar == 100.0``.
    nan_policy : str, default 'ignore'
        Passed to the internal EMAs.

    Returns
    -------
    tuple of np.ndarray
        (ppo_line, signal_line, histogram).

    Raises
    ------
    ValueError
        If ``fast``, ``slow`` or ``signal`` < 1.

    """
    if fast < 1:
        raise ValueError('fast must be >= 1')
    if slow < 1:
        raise ValueError('slow must be >= 1')
    if signal < 1:
        raise ValueError('signal must be >= 1')
    close = np.asarray(close, dtype=np.float64, copy=False)
    if close.size == 0:
        empty = np.array([])
        return empty, empty, empty
    if close.size < min(fast, slow):
        nan_arr = np.full(close.size, np.nan)
        return nan_arr, nan_arr.copy(), nan_arr.copy()
    close = close.copy()
    replace_inf_with_nan(close)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # ---- PPO line ----
    if use_talib and talib_available and scalar == 100.0:
        ppo_line = talib.PPO(close, fast, slow, 0)
    else:
        fast_ema = ema_ind(
            close, length=fast, use_talib=False, nan_policy=nan_policy
        )
        slow_ema = ema_ind(
            close, length=slow, use_talib=False, nan_policy=nan_policy
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            ppo_line = scalar * (fast_ema - slow_ema) / slow_ema
    # ---- Signal line (same logic as macd_numpy) ----
    # Forward-fill the warm-up NaN prefix with the first valid value so
    # the EMA recursion starts cleanly, then mask the standard prefix.
    ppo_filled = ppo_line.copy()
    first_valid = np.argmax(~np.isnan(ppo_line))
    if not np.isnan(ppo_line[first_valid]):
        ppo_filled[:first_valid] = ppo_line[first_valid]
    signalma = ema_ind(
        ppo_filled, length=signal, use_talib=False, nan_policy='ignore'
    )
    signalma[:slow + signal - 2] = np.nan
    hist = ppo_line - signalma
    # ---- Apply offset and fillna ----
    ppo_line = _apply_offset_fillna(ppo_line, offset, fillna)
    signalma = _apply_offset_fillna(signalma, offset, fillna)
    hist = _apply_offset_fillna(hist, offset, fillna)
    return ppo_line, signalma, hist


def ppo_ind(
    close: np.ndarray | pl.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'ignore',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Universal wrapper for PPO that accepts either a NumPy array or a
    Polars Series. All parameters are the same as in ``ppo_numpy``.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return ppo_numpy(
        close, fast, slow, signal, scalar,
        offset, fillna, use_talib, nan_policy,
    )


def ppo_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    scalar: float = 100.0,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'ignore',
    suffix: str = '',
) -> pl.DataFrame:
    """Add PPO columns to a Polars DataFrame.

    Added columns (names depend on ``suffix``):
        - ``PPO{suffix}_{fast}_{slow}_{signal}``  (line)
        - ``PPOs{suffix}_{fast}_{slow}_{signal}`` (signal)
        - ``PPOh{suffix}_{fast}_{slow}_{signal}`` (histogram)
    """
    close = df[close_col].cast(pl.Float64).to_numpy()
    ppo_line, signal_line, hist = ppo_numpy(
        close, fast, slow, signal, scalar,
        offset, fillna, use_talib, nan_policy,
    )
    if not suffix:
        suffix = f'_{fast}_{slow}_{signal}'
    return df.with_columns([
        pl.Series(f'PPO{suffix}', ppo_line),
        pl.Series(f'PPOs{suffix}', signal_line),
        pl.Series(f'PPOh{suffix}', hist),
    ])
