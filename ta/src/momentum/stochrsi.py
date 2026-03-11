# -*- coding: utf-8 -*-
from typing import cast

import numpy as np
import polars as pl

from ..ma import ma_mode
from ..momentum import rsi_ind
from ..utils import (
    _apply_offset_fillna,
    _handle_nan_policy,
    _rolling_max_numba,
    _rolling_min_numba,
)


# ----------------------------------------------------------------------
# Core Stochastic RSI calculation (numpy)
# ----------------------------------------------------------------------
def stochrsi_numpy(
    close: np.ndarray,
    length: int = 14,
    rsi_length: int = 14,
    k: int = 3,
    d: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numpy-based Stochastic RSI calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    length : int
        Period for rolling min/max of RSI.
    rsi_length : int
        Period for RSI calculation.
    k : int
        Fast %K period (moving average of stoch).
    d : int
        Slow %D period (moving average of %K).
    mamode : str
        Moving average mode for %K and %D smoothing.
    offset, fillna, use_talib, nan_policy, trim : as usual.

    Returns
    -------
    (stoch_k, stoch_d) as numpy arrays.
    """
    # Input validation
    if length < 1 or rsi_length < 1 or k < 1 or d < 1:
        raise ValueError("All period lengths must be >= 1")
    close = np.asarray(close, dtype=np.float64)
    # Check for inf
    if np.isinf(close).any():
        raise ValueError("Input contains non-finite values (inf or -inf).")
    # Handle NaN
    close = _handle_nan_policy(close, nan_policy, "close")
    # Ensure contiguous
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # 1. RSI
    rsi = cast(np.ndarray, rsi_ind(
        close, length=rsi_length, use_talib=use_talib, nan_policy=nan_policy
    ))
    # 2. Rolling min and max of RSI
    lowest_rsi = _rolling_min_numba(rsi, length)
    highest_rsi = _rolling_max_numba(rsi, length)
    # 3. Stochastic
    denom = highest_rsi - lowest_rsi
    with np.errstate(divide='ignore', invalid='ignore'):
        stoch = np.where(denom != 0, 100.0 * (rsi - lowest_rsi) / denom, 50.0)
        # If denom == 0, we set stoch to 50 (neutral) to avoid division by zero
    # 4. Smooth to get %K and %D
    stoch_k = cast(np.ndarray, ma_mode(
        mamode, stoch, length=k, use_talib=use_talib, nan_policy=nan_policy
    ))
    stoch_d = cast(np.ndarray, ma_mode(
        mamode, stoch_k, length=d, use_talib=use_talib, nan_policy=nan_policy
    ))
    # 5. Trim if requested
    if trim:
        # Minimum required length: RSI needs rsi_length, 
        # then we need length for min/max,
        # then k and d for smoothing. The first valid value of %D appears at index:
        # rsi_length - 1 + (length - 1) + (k - 1) + (d - 1) = rsi_length + length + k + d - 4  # noqa: E501
        # But we'll use a simpler conservative approach: trim = True returns only values
        # where all components are defined. Usually this is the 
        # last `len - (rsi_length + length + k + d - 4)`.
        # However, to keep consistent with other indicators, we'll just return from
        # the index where stoch_d is first non-NaN.
        first_valid = np.where(~np.isnan(stoch_d))[0]
        if len(first_valid) > 0:
            start = first_valid[0]
            stoch_k = stoch_k[start:]
            stoch_d = stoch_d[start:]
        else:
            # No valid data, return empty arrays
            stoch_k = np.array([])
            stoch_d = np.array([])
    # 6. Apply offset and fillna
    stoch_k = _apply_offset_fillna(stoch_k, offset, fillna)
    stoch_d = _apply_offset_fillna(stoch_d, offset, fillna)
    return stoch_k, stoch_d


# ----------------------------------------------------------------------
# Universal Stochastic RSI (accepts numpy or Polars Series)
# ----------------------------------------------------------------------
def stochrsi_ind(
    close: np.ndarray | pl.Series,
    length: int = 14,
    rsi_length: int = 14,
    k: int = 3,
    d: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal Stochastic RSI (accepts numpy array or Polars Series).
    Returns (stoch_k, stoch_d) as numpy arrays.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return stochrsi_numpy(
        close,
        length=length,
        rsi_length=rsi_length,
        k=k,
        d=d,
        mamode=mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=trim,
    )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def stochrsi_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 14,
    rsi_length: int = 14,
    k: int = 3,
    d: int = 3,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col_k: str | None = None,
    output_col_d: str | None = None,
) -> pl.DataFrame:
    """
    Add Stochastic RSI %K and %D columns to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Column with close prices.
    length, rsi_length, k, d, mamode, offset, fillna, use_talib, nan_policy : as above.
    output_col_k : str, optional
        Output column name for %K (default "STOCHRSIk_{length}_{rsi_length}_{k}_{d}").
    output_col_d : str, optional
        Output column name for %D (default "STOCHRSId_{length}_{rsi_length}_{k}_{d}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added columns (same length).
    """
    close = df[close_col].to_numpy()
    stoch_k, stoch_d = stochrsi_numpy(
        close,
        length=length,
        rsi_length=rsi_length,
        k=k,
        d=d,
        mamode=mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars always returns full length
    )
    # Generate default column names
    if output_col_k is None:
        output_col_k = f"STOCHRSIk_{length}_{rsi_length}_{k}_{d}"
    if output_col_d is None:
        output_col_d = f"STOCHRSId_{length}_{rsi_length}_{k}_{d}"
    return df.with_columns([
        pl.Series(output_col_k, stoch_k),
        pl.Series(output_col_d, stoch_d),
    ])