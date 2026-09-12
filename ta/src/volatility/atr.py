# -*- coding: utf-8 -*-
"""Average True Range (ATR) implementation.

ATR smooths the True Range with a moving average (Wilder RMA by
default; SMA and EMA modes are also supported).  The True Range
warm-up (first ``drift`` values are NaN) is cut before smoothing, so
the full ATR warm-up is ``drift + length - 1`` bars.

The module provides:
- ``atr_numba`` - TR (Numba) + smoothing with NaN handling, trim
- ``atr_talib`` - TA-Lib backend (Wilder RMA, drift=1)
- ``atr_ind`` - universal wrapper with automatic backend selection
  (``mamode != 'rma'`` or ``drift != 1`` force the Numba backend)
- ``atr_polars`` - Polars DataFrame integration

All floating-point operations follow IEEE 754 rules.  Infinite values
are replaced with NaN before calculation.
"""
import numpy as np
import polars as pl

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..external import talib, talib_available
from ..overlap.ema import ema_ind
from ..overlap.rma import rma_ind
from ..overlap.sma import sma_ind
from .true_range import true_range_numba


# ----------------------------------------------------------------------
# Shared validation / preprocessing
# ----------------------------------------------------------------------
def _validate_atr_inputs(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int,
    drift: int,
) -> None:
    """Validate ATR inputs; raise ValueError on inconsistency.

    Args:
        high: High price array.
        low: Low price array.
        close: Close price array.
        length: ATR smoothing period (must be >= 1).
        drift: True Range drift (must be >= 1).

    Raises:
    ------
    ValueError
        If ``length < 1``, ``drift < 1``, the arrays have different
        lengths, or the series is too short (need at least
        ``drift + length`` elements for a non-empty ATR).

    """
    if length < 1:
        raise ValueError(f'ATR length must be >= 1, got {length}')
    if drift < 1:
        raise ValueError(
            f'ATR drift must be >= 1 (got {drift}); drift < 1 would '
            'either compare a bar with itself or look into the future.'
        )
    if not (len(high) == len(low) == len(close)):
        raise ValueError(
            f'high, low and close must have the same length: '
            f'got {len(high)}, {len(low)} and {len(close)}.'
        )
    if len(high) < drift + length:
        raise ValueError(
            f'Input series too short: need at least {drift + length} '
            f'elements, got {len(high)}.'
        )


def _prepare_prices(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    nan_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast to float64, replace infinities with NaN, apply the policy."""
    high = np.ascontiguousarray(np.asarray(high, dtype=np.float64))
    low = np.ascontiguousarray(np.asarray(low, dtype=np.float64))
    close = np.ascontiguousarray(np.asarray(close, dtype=np.float64))
    high = high.copy()
    low = low.copy()
    close = close.copy()
    replace_inf_with_nan(high)
    replace_inf_with_nan(low)
    replace_inf_with_nan(close)
    high = _handle_nan_policy(high, nan_policy, 'high')
    low = _handle_nan_policy(low, nan_policy, 'low')
    close = _handle_nan_policy(close, nan_policy, 'close')
    return high, low, close


# ----------------------------------------------------------------------
# ATR – Numba implementation (TR + RMA/SMA/EMA) with NaN handling and trim
# ----------------------------------------------------------------------
def atr_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    mamode: str = 'rma',
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Average True Range using Numba for TR and smoothing.

    The True Range warm-up (the first ``drift`` values are NaN) is cut
    before smoothing, so the NaN policy of the smoothing MA is never
    triggered by the warm-up itself.  The full ATR warm-up is
    ``drift + length - 1`` bars.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays, same length.
    length : int
        ATR period (must be >= 1).
    mamode : str
        Moving average mode ('rma', 'sma', 'ema').
    drift : int
        Lookback for the True Range (must be >= 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    percent : bool
        If True, return ATR as a percentage of close.
    nan_policy : str, default 'raise'
        How to handle NaN values in the price columns.
    trim : bool
        If True, remove the ``drift + length - 1`` warm-up values.

    Returns
    -------
    np.ndarray
        ATR values.

    """
    mamode_lower = mamode.lower()
    if mamode_lower not in ('rma', 'sma', 'ema'):
        raise ValueError(f'Unsupported mamode: {mamode}')
    _validate_atr_inputs(high, low, close, length, drift)
    high, low, close = _prepare_prices(high, low, close, nan_policy)

    # True Range; prices are already policy-clean, so only the TR
    # warm-up produces NaN here - it is cut below before smoothing.
    tr = true_range_numba(
        high, low, close, drift=drift, nan_policy='ignore'
    )
    tr_valid = tr[drift:]

    if mamode_lower == 'rma':
        ma_part = rma_ind(tr_valid, length, nan_policy='ignore')
    elif mamode_lower == 'sma':
        ma_part = sma_ind(tr_valid, length, nan_policy='ignore')
    else:  # 'ema'
        ma_part = ema_ind(tr_valid, length, nan_policy='ignore')

    n = len(high)
    atr = np.full(n, np.nan, dtype=np.float64)
    atr[drift:] = ma_part

    # Convert to percent if requested
    if percent:
        with np.errstate(divide='ignore', invalid='ignore'):
            atr = atr * 100.0 / close
    # Trim the full warm-up (drift + length - 1 bars)
    if trim:
        start = drift + length - 1
        atr = atr[start:]
    # Apply offset and fillna
    return _apply_offset_fillna(atr, offset, fillna)


# ----------------------------------------------------------------------
# ATR – TA-Lib wrapper with NaN handling and trim
# ----------------------------------------------------------------------
def atr_talib(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """ATR using TA-Lib (Wilder RMA, drift=1) with pre‑processing.

    Parameters
    ----------
    high, low, close : np.ndarray
        Price arrays, same length.
    length : int
        ATR period (must be >= 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    percent : bool
        If True, return ATR as a percentage of close.
    nan_policy : str, default 'raise'
        How to handle NaN values in the price columns.
    trim : bool
        If True, remove the ``length`` warm-up values.

    Returns
    -------
    np.ndarray
        ATR values.

    """
    if not talib_available:
        raise ImportError('TA-Lib is not available')
    _validate_atr_inputs(high, low, close, length, drift=1)
    high, low, close = _prepare_prices(high, low, close, nan_policy)
    atr = talib.ATR(high, low, close, timeperiod=length)
    if percent:
        with np.errstate(divide='ignore', invalid='ignore'):
            atr = atr * 100.0 / close
    if trim:
        # TA-Lib ATR has `length` leading NaNs (its own warm-up)
        atr = atr[length:]
    return _apply_offset_fillna(atr, offset, fillna)


# ----------------------------------------------------------------------
# Universal ATR function
# ----------------------------------------------------------------------
def atr_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 14,
    mamode: str = 'rma',
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """Universal Average True Range with automatic backend selection.

    Backend rules:
    - ``mamode != 'rma'`` or ``drift != 1`` **force the Numba
        backend**: TA-Lib ATR is hard-wired to Wilder RMA with
        drift=1 and would silently ignore those parameters.
    - Otherwise TA-Lib is used when ``use_talib=True`` and available.

    Parameters
    ----------
    high, low, close : np.ndarray or pl.Series
        Price series, same length.
    length : int
        ATR period (must be >= 1).
    mamode : str
        Moving average mode ('rma', 'sma', 'ema').
    drift : int
        True Range drift (must be >= 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    percent : bool
        If True, return ATR as a percentage of close.
    use_talib : bool
        Use TA-Lib if available (rma mode, drift=1 only).
    nan_policy : str, default 'raise'
        How to handle NaN values in the price columns.
    trim : bool
        If True, remove the warm-up values.

    Returns
    -------
    np.ndarray
        ATR values.

    """
    # Convert Polars Series to numpy
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    talib_can_honour = mamode.lower() == 'rma' and drift == 1
    if use_talib and talib_available and talib_can_honour:
        return atr_talib(
            high, low, close,
            length=length,
            offset=offset,
            fillna=fillna,
            percent=percent,
            nan_policy=nan_policy,
            trim=trim,
        )
    return atr_numba(
        high, low, close,
        length=length,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        percent=percent,
        nan_policy=nan_policy,
        trim=trim,
    )


# ----------------------------------------------------------------------
# Polars integration for ATR (always full length)
# ----------------------------------------------------------------------
def atr_polars(
    df: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    length: int = 14,
    mamode: str = 'rma',
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    percent: bool = False,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None
) -> pl.DataFrame:
    """ATR for Polars DataFrame (returns same length, no trim).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    high_col, low_col, close_col : str
        Column names for prices.
    length : int
        ATR period (must be >= 1).
    mamode : str
        Moving average mode ('rma', 'sma', 'ema').
    drift : int
        True Range drift (must be >= 1; mamode != 'rma' or drift != 1
        forces the Numba backend).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    percent : bool
        If True, return ATR as a percentage of close.
    use_talib : bool
        Use TA-Lib if available (rma mode, drift=1 only).
    nan_policy : str, default 'raise'
        How to handle NaN values in the price columns.
    output_col : str, optional
        Output column name (default f"ATR_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added column.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    result = atr_ind(
        high, low, close,
        length=length,
        mamode=mamode,
        drift=drift,
        offset=offset,
        fillna=fillna,
        percent=percent,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars always returns full length
    )
    out_name = output_col or f'ATR_{length}'
    return df.with_columns([pl.Series(out_name, result)])
