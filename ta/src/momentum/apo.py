# -*- coding: utf-8 -*-
from typing import cast

import numpy as np
import polars as pl

from .. import ma_mode
from ..utils import _apply_offset_fillna


def apo_numpy(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based APO calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    fast : int
        Fast MA period.
    slow : int
        Slow MA period.
    mamode : str
        Moving average type (passed to `ma`).
    offset, fillna, use_talib : as usual.

    Returns
    -------
    np.ndarray
        APO values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # Ensure slow is the larger period (original behaviour)
    if slow < fast:
        fast, slow = slow, fast
    fast_ma = cast(np.ndarray, ma_mode(
        mamode, close, length=fast, offset=0, fillna=None, use_talib=use_talib))
    slow_ma = cast(np.ndarray, ma_mode(
        mamode, close, length=slow, offset=0, fillna=None, use_talib=use_talib))
    apo = fast_ma - slow_ma
    return _apply_offset_fillna(apo, offset, fillna)


def apo_ind(
    close: np.ndarray | pl.Series,
    fast: int = 12,
    slow: int = 26,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Absolute Price Oscillator (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return apo_numpy(close, fast, slow, mamode, offset, fillna, use_talib)


def apo_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    fast: int = 12,
    slow: int = 26,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    fast, slow, mamode, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"APO_{fast}_{slow}").

    Returns
    -------
    pl.DataFrame
    """
    close = df[close_col].to_numpy()
    result = apo_ind(close, fast, slow, mamode, offset, fillna, use_talib)
    out_name = output_col or f"APO_{fast}_{slow}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })