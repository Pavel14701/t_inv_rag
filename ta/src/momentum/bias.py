# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import ma_mode
from ..utils import _apply_offset_fillna


def bias_numpy(
    close: np.ndarray,
    length: int = 26,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based Bias calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        MA period.
    mamode : str
        Moving average type.
    offset, fillna, use_talib : as usual.

    Returns
    -------
    np.ndarray
        Bias values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    ma = ma_mode(
        mamode, close, length=length, offset=0, fillna=None, use_talib=use_talib
    )
    bias = (close / ma) - 1.0
    return _apply_offset_fillna(bias, offset, fillna)


def bias_ind(
    close: np.ndarray | pl.Series,
    length: int = 26,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Bias (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return bias_numpy(close, length, mamode, offset, fillna, use_talib)


def bias_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    length: int = 26,
    mamode: str = "sma",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add Bias column to Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Column with close prices.
    length, mamode, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"BIAS_{mamode}_{length}").
    Returns
    -------
    pl.DataFrame
    """
    close = df[close_col].to_numpy()
    result = bias_ind(close, length, mamode, offset, fillna, use_talib)
    out_name = output_col or f"BIAS_{mamode}_{length}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })