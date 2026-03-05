# -*- coding: utf-8 -*-
from typing import cast

import numpy as np
import polars as pl

from ..ma import ma_mode
from ..utils import _apply_offset_fillna


def accbands_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 20,
    c: float = 4.0,
    mamode: str = "sma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy‑based Acceleration Bands calculation.
    Returns (upper, mid, lower) as numpy arrays.
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    for arr in (high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    high_low_range = high - low
    hl_ratio = high_low_range / (high + low) * c
    lower_raw = low * (1.0 - hl_ratio)
    upper_raw = high * (1.0 + hl_ratio)
    lower = cast(np.ndarray, ma_mode(
        mamode, lower_raw,
        length=length, offset=0, fillna=None, use_talib=use_talib
    ))
    mid = cast(np.ndarray, ma_mode(
        mamode, close,
        length=length, offset=0, fillna=None, use_talib=use_talib
    ))
    upper = cast(np.ndarray, ma_mode(
        mamode, upper_raw,
        length=length, offset=0, fillna=None, use_talib=use_talib
    ))
    lower = _apply_offset_fillna(lower, offset, fillna)
    mid = _apply_offset_fillna(mid, offset, fillna)
    upper = _apply_offset_fillna(upper, offset, fillna)
    return upper, mid, lower


def accbands(
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 20,
    c: float = 4.0,
    mamode: str = "sma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal Acceleration Bands (accepts numpy arrays or Polars Series).
    Returns (upper, mid, lower) as numpy arrays.
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return accbands_numpy(
        high, low, close, length, c, mamode, drift, offset, fillna, use_talib
    )


def accbands_polars(
    df: pl.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    length: int = 20,
    c: float = 4.0,
    mamode: str = "sma",
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add Acceleration Bands columns to Polars DataFrame.

    Columns added:
        ACCBU_{length}   (upper band)
        ACCBM_{length}   (middle band)
        ACCBL_{length}   (lower band)

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    high_col, low_col, close_col : str
        Column names for prices.
    length, c, mamode, drift, offset, fillna, use_talib : as above.
    suffix : str
        Custom suffix for column names (default f"_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with three new columns.
    """
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    upper, mid, lower = accbands_numpy(
        high, low, close, length, c, mamode, drift, offset, fillna, use_talib
    )
    suffix = suffix or f"_{length}"
    return df.with_columns([
        pl.Series(f"ACCBU{suffix}", upper),
        pl.Series(f"ACCBM{suffix}", mid),
        pl.Series(f"ACCBL{suffix}", lower),
    ])