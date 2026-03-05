# -*- coding: utf-8 -*-
"""
Bollinger Bands (BBANDS) – Numba‑accelerated with Polars integration.
"""

from typing import Optional, cast

import numpy as np
import polars as pl

from ..ma import ma_mode
from ..statistics import stdev_ind
from ..utils import _apply_offset_fillna


def bbands_numpy(
    close: np.ndarray,
    length: int = 20,
    lower_std: float = 2.0,
    upper_std: float = 2.0,
    ddof: int = 1,
    mamode: str = "sma",
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy‑based Bollinger Bands calculation.

    Returns (lower, mid, upper, bandwidth, percent_b) as numpy arrays.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    mid = cast(np.ndarray, ma_mode(
            mamode, close, length=length, offset=0, fillna=None, use_talib=use_talib
    ))
    std = stdev_ind(close, length=length, ddof=ddof, use_talib=use_talib)
    lower_deviations = lower_std * std
    upper_deviations = upper_std * std
    lower = mid - lower_deviations
    upper = mid + upper_deviations
    # Bandwidth and %B
    ulr = upper - lower
    bandwidth = 100.0 * ulr / mid
    percent_b = (close - lower) / ulr
    # Apply offset and fillna to all five series
    lower = _apply_offset_fillna(lower, offset, fillna)
    mid = _apply_offset_fillna(mid, offset, fillna)
    upper = _apply_offset_fillna(upper, offset, fillna)
    bandwidth = _apply_offset_fillna(bandwidth, offset, fillna)
    percent_b = _apply_offset_fillna(percent_b, offset, fillna)
    return lower, mid, upper, bandwidth, percent_b


def bbands(
    close: np.ndarray | pl.Series,
    length: int = 20,
    lower_std: float = 2.0,
    upper_std: float = 2.0,
    ddof: int = 1,
    mamode: str = "sma",
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal Bollinger Bands (accepts numpy array or Polars Series).

    Returns (lower, mid, upper, bandwidth, percent_b) as numpy arrays.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return bbands_numpy(
        close,
        length=length,
        lower_std=lower_std,
        upper_std=upper_std,
        ddof=ddof,
        mamode=mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )


def bbands_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 20,
    lower_std: float = 2.0,
    upper_std: float = 2.0,
    ddof: int = 1,
    mamode: str = "sma",
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add Bollinger Bands columns to Polars DataFrame.

    Columns added:
        BBL_{length}_{lower_std}_{upper_std}
        BBM_{length}_{lower_std}_{upper_std}
        BBU_{length}_{lower_std}_{upper_std}
        BBB_{length}_{lower_std}_{upper_std}
        BBP_{length}_{lower_std}_{upper_std}

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, lower_std, upper_std, ddof, mamode, offset, fillna, use_talib : as above.
    suffix : str
        Custom suffix for column names (default f"_{length}_{lower_std}_{upper_std}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with five new columns.
    """
    close = df[close_col].to_numpy()
    lower, mid, upper, bandwidth, percent_b = bbands_numpy(
        close,
        length=length,
        lower_std=lower_std,
        upper_std=upper_std,
        ddof=ddof,
        mamode=mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )

    suffix = suffix or f"_{length}_{lower_std}_{upper_std}"
    return df.with_columns([
        pl.Series(f"BBL{suffix}", lower),
        pl.Series(f"BBM{suffix}", mid),
        pl.Series(f"BBU{suffix}", upper),
        pl.Series(f"BBB{suffix}", bandwidth),
        pl.Series(f"BBP{suffix}", percent_b),
    ])