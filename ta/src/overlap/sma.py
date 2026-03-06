# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import float64, int64, njit

from .. import talib, talib_available


# ----------------------------------------------------------------------
# Optimized SMA using Numba (nopython mode, fastmath) with type signature
# ----------------------------------------------------------------------
@njit((float64[:], int64), fastmath=True, cache=True)
def _sma_numba_opt(arr: np.ndarray, length: int) -> np.ndarray:
    """
    Simple Moving Average using Numba.

    Parameters
    ----------
    arr : np.ndarray
        1D float64 array without NaNs (should be preprocessed).
    length : int
        Window length.

    Returns
    -------
    np.ndarray
        SMA values; first (length-1) positions are NaN.
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < length:
        return out
    cum = np.cumsum(arr)
    out[length - 1:] = (
        cum[length - 1:] - np.concatenate(([0], cum[:n - length]))
    ) / length
    return out


# ----------------------------------------------------------------------
# SMA using Numba (with offset, fillna, NaN handling, and trim)
# ----------------------------------------------------------------------
def _sma_numba(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """
    Simple Moving Average using Numba with NaN handling.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        SMA period (>= 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs in the result.
    nan_policy : str, default 'raise'
        How to handle NaNs in the input:
        - 'raise': raise ValueError if any NaN is present.
        - 'ffill': forward fill (propagate last valid observation).
        - 'bfill': backward fill (propagate next valid observation).
        - 'both': first forward fill, then backward fill (fills all gaps).
    trim : bool, default False
        If True, return only the valid part of SMA (first `length-1` elements removed).
        The output length becomes `len(close) - (length-1)`.

    Returns
    -------
    np.ndarray
        SMA values.
    """
    # ---- Input validation ----
    if length < 1:
        raise ValueError("SMA length must be >= 1")
    close = np.asarray(close, dtype=np.float64)
    # Check for infinite values (they would break calculations)
    if np.isinf(close).any():
        raise ValueError("Input contains non-finite values (inf or -inf).")
    # ---- NaN handling on input ----
    if np.isnan(close).any():
        if nan_policy == 'raise':
            raise ValueError("Input contains NaN values. Use nan_policy='ffill', \
                'bfill' or 'both' to fill them.")
        elif nan_policy == 'ffill':
            close = close.copy()
            for i in range(1, len(close)):
                if np.isnan(close[i]):
                    close[i] = close[i - 1]
        elif nan_policy == 'bfill':
            close = close.copy()
            for i in range(len(close) - 2, -1, -1):
                if np.isnan(close[i]):
                    close[i] = close[i + 1]
        elif nan_policy == 'both':
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
    sma = _sma_numba_opt(close, length)
    # Trim if requested (remove initial NaNs)
    if trim:
        sma = sma[length - 1:]
    # Apply offset (only if not trimmed, 
    # because offset would then be applied to shorter array)
    # But offset is applied after trim? Usually offset is a shift on the original index.
    # If we trim, offset should be applied relative to the new shorter array.
    # For simplicity, we apply offset before trim? But then offset might shift 
    # NaNs into valid region.
    # Better: apply offset to the full-length array, then trim.
    # However, offset is typically applied after calculation to align with 
    # original index.
    # If user wants trim=True, they likely want the valid values without any offset.
    # We'll document that offset and trim are not compatible 
    # and raise error if both used.
    if offset != 0 and trim:
        raise ValueError(
            "offset and trim cannot be used simultaneously. \
                Use offset=0 with trim=True."
        )
    if offset != 0:
        sma = np.roll(sma, offset)
        if offset > 0:
            sma[:offset] = np.nan
        else:
            sma[offset:] = np.nan
    if fillna is not None:
        sma = np.where(np.isnan(sma), fillna, sma)
    return sma


# ----------------------------------------------------------------------
# SMA using TA-Lib (unchanged, but could also be extended with nan_policy if needed)
# ----------------------------------------------------------------------
def sma_talib(
    close: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Simple Moving Average via TA-Lib.
    (Note: TA-Lib does not handle NaNs; it's assumed input is clean.)
    """
    if not talib_available:
        raise ImportError("TA-Lib is not available")
    close = close.astype(np.float64)
    sma = talib.SMA(close, timeperiod=length)
    if offset != 0:
        sma = np.roll(sma, offset)
        if offset > 0:
            sma[:offset] = np.nan
        else:
            sma[offset:] = np.nan
    if fillna is not None:
        sma = np.where(np.isnan(sma), fillna, sma)
    return sma


# ----------------------------------------------------------------------
# Universal SMA function (automatic backend selection) with nan_policy and trim
# ----------------------------------------------------------------------
def sma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    trim: bool = False,
) -> np.ndarray:
    """
    Universal SMA with automatic implementation selection and NaN handling.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        SMA period.
    offset : int
        Shift result (incompatible with trim=True).
    fillna : float, optional
        Fill NaN with this value.
    use_talib : bool
        If True and TA-Lib is available, use it; else use Numba.
    nan_policy : str, default 'raise'
        How to handle NaNs in the input 
        (only for Numba backend; TA-Lib assumes clean data).
    trim : bool, default False
        If True and using Numba backend, return only 
        valid part (first `length-1` values removed).
        TA-Lib backend does not support trim (raises error if trim=True).

    Returns
    -------
    np.ndarray
        SMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = close.astype(np.float64)

    if use_talib and talib_available:
        if trim:
            raise ValueError(
                "trim=True is not supported with TA-Lib backend. Use Numba backend."
            )
        return sma_talib(close, length, offset, fillna)
    else:
        return _sma_numba(close, length, offset, fillna, nan_policy, trim)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def sma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = 'raise',
    output_col: str | None = None
) -> pl.DataFrame:
    """
    SMA for Polars DataFrame with NaN handling.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    length : int
        SMA period.
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    use_talib : bool
        Use TA-Lib if available.
    nan_policy : str, default 'raise'
        How to handle NaNs in the input (for Numba backend).
    output_col : str, optional
        Output column name (default f"SMA_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with added column (same length).
    """
    close = df[close_col].to_numpy()
    # Polars version always returns full-length array (trim=False is implicit)
    result = sma_ind(
        close,
        length=length,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,
    )
    out_name = output_col or f"SMA_{length}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })