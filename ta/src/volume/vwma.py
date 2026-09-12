# -*- coding: utf-8 -*-
"""Volume Weighted Moving Average (VWMA) implementation.

VWMA weights each close price by its volume over a rolling window:
``VWMA = SMA(close * volume) / SMA(volume)``.  Both SMAs are delegated
to :func:`ta.src.overlap.sma.sma_ind` (Numba or TA-Lib backend).

The module provides:
- ``vwma_numpy`` - numpy-based calculation with NaN handling
- ``vwma_ind`` - universal wrapper (numpy arrays or Polars Series)
- ``vwma_polars`` - Polars DataFrame integration

All floating-point operations follow IEEE 754 rules.  Infinite values
are replaced with NaN before calculation.  Windows whose total volume
is zero produce NaN (0/0).
"""

import numpy as np
import polars as pl

from .._array_ops import (
    _apply_offset_fillna,
    _handle_nan_policy,
    replace_inf_with_nan,
)
from ..overlap.sma import sma_ind


def vwma_numpy(
    close: np.ndarray,
    volume: np.ndarray,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Numpy-based VWMA calculation.

    Parameters
    ----------
    close, volume : np.ndarray
        Price and volume arrays (float64), same length.
    length : int
        VWMA period (must be >= 1).
    offset : int
        Shift applied to the result.
    fillna : float, optional
        Value to fill NaNs and shifted-in positions.
    use_talib : bool
        If True and TA-Lib is installed, the SMA inside is computed
        via TA-Lib; otherwise via the Numba backend.
    nan_policy : str, default 'raise'
        How to handle NaN values in ``close``/``volume``:
        'raise', 'ignore', 'ffill', 'bfill', or 'both'.  The policy is
        applied before any backend is used, so both backends behave
        identically.

    Returns
    -------
    np.ndarray
        VWMA values; the first ``length - 1`` positions are NaN (or
        ``fillna``).  Windows whose total volume is zero yield NaN.

    """
    if length < 1:
        raise ValueError(f"VWMA length must be >= 1, got {length}")
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    if len(close) != len(volume):
        raise ValueError(
            f"close and volume must have the same length: "
            f"got {len(close)} and {len(volume)}."
        )
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not volume.flags.c_contiguous:
        volume = np.ascontiguousarray(volume)

    # Replace infinities with NaN (IEEE 754 compliance) so that the
    # chosen nan_policy applies to them as well.
    close = close.copy()
    volume = volume.copy()
    replace_inf_with_nan(close)
    replace_inf_with_nan(volume)

    # Validate the policy (always) and handle NaN before the backend
    # runs.  This keeps 'raise'/'ffill'/... behaviour consistent and
    # rejects unknown policies even when no NaN is present.
    close = _handle_nan_policy(close, nan_policy, "close")
    volume = _handle_nan_policy(volume, nan_policy, "volume")

    if len(close) < length:
        raise ValueError(
            f"Input series too short: need at least {length} elements, "
            f"got {len(close)}."
        )

    # Price * volume
    pv = close * volume
    # SMA of pv and volume; the nan_policy is forwarded so that any
    # remaining NaN (e.g. with 'ignore') is respected by the backend.
    sma_pv = sma_ind(
        pv,
        length=length,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )
    sma_vol = sma_ind(
        volume,
        length=length,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )
    # VWMA = SMA(pv) / SMA(vol); windows with zero total volume -> NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        vwma = sma_pv / sma_vol
    return _apply_offset_fillna(vwma, offset, fillna)


def vwma_ind(
    close: np.ndarray | pl.Series,
    volume: np.ndarray | pl.Series,
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
) -> np.ndarray:
    """Universal VWMA (accepts numpy arrays or Polars Series).

    Parameters
    ----------
    close, volume : np.ndarray or pl.Series
        Price and volume series, same length.
    length : int
        VWMA period (must be >= 1).
    offset, fillna, use_talib, nan_policy
        As in :func:`vwma_numpy`.

    Returns
    -------
    np.ndarray
        VWMA values.

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if isinstance(volume, pl.Series):
        volume = volume.to_numpy()
    return vwma_numpy(
        close, volume, length, offset, fillna, use_talib, nan_policy
    )


def vwma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    length: int = 10,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    output_col: str | None = None,
) -> pl.DataFrame:
    """Add VWMA column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col, volume_col : str
        Column names for prices and volume.
    length : int
        VWMA period (must be >= 1).
    offset : int
        Shift applied to the result.
    fillna : float, optional
        Value to fill NaNs and shifted-in positions.
    use_talib : bool
        If True and TA-Lib is installed, the SMA inside is computed
        via TA-Lib; otherwise via the Numba backend.
    nan_policy : str, default 'raise'
        How to handle NaN values in the close/volume columns.
    output_col : str, optional
        Output column name (default f"VWMA_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new column.

    Notes
    -----
    - The function does not modify the original DataFrame in-place.
    - All operations are IEEE 754 compliant.

    """
    close = df[close_col].to_numpy()
    volume = df[volume_col].to_numpy()
    result = vwma_numpy(
        close, volume, length, offset, fillna, use_talib, nan_policy
    )
    out_name = output_col or f"VWMA_{length}"
    return df.with_columns([pl.Series(out_name, result)])
