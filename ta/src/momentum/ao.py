# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna
from ..overlap.sma import sma_ind


def ao_numpy(
    high: np.ndarray,
    low: np.ndarray,
    fast: int = 5,
    slow: int = 34,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Numpy-based Awesome Oscillator calculation.

    Parameters
    ----------
    high, low : np.ndarray
        Price arrays (float64).
    fast : int
        Fast SMA period.
    slow : int
        Slow SMA period.
    offset, fillna, use_talib : as usual.

    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.
    use_talib : bool, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    np.ndarray
        AO values.

    Raises
    ------
    ValueError
        If `fast` < 1 or `slow` < 1.

    """
    if fast < 1:
        raise ValueError("fast must be >= 1")
    if slow < 1:
        raise ValueError("slow must be >= 1")
    # Ensure arrays are contiguous
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    # Rebind the outer names: assigning to the loop variable is a no-op
    # and left the arrays non-contiguous.
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    # Swap if slow < fast (original behaviour)
    if slow < fast:
        fast, slow = slow, fast
    median = (high + low) * 0.5
    fast_sma = sma_ind(
        median, length=fast, offset=0, fillna=None, use_talib=use_talib
    )
    slow_sma = sma_ind(
        median, length=slow, offset=0, fillna=None, use_talib=use_talib
    )
    ao = fast_sma - slow_sma
    return _apply_offset_fillna(ao, offset, fillna)


def ao_ind(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    fast: int = 5,
    slow: int = 34,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Universal Awesome Oscillator (accepts numpy arrays or Polars Series)."""
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    return ao_numpy(high, low, fast, slow, offset, fillna, use_talib)


def ao_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    date_col: str = "date",
    fast: int = 5,
    slow: int = 34,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col : str
        Column names for high and low prices.
    fast, slow, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"AO_{fast}_{slow}").

    Returns
    -------
    pl.DataFrame

    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    result = ao_numpy(high, low, fast, slow, offset, fillna, use_talib)
    out_name = output_col or f"AO_{fast}_{slow}"
    return pl.DataFrame({date_col: df[date_col], out_name: result})
