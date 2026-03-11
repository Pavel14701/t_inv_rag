# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..overlap import sma_ind
from ..utils import _apply_offset_fillna
from . import stdev_ind


def zscore_numpy(
    close: np.ndarray,
    length: int = 30,
    multiplier: float = 1.0,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based rolling Z‑Score.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Window length.
    multiplier : float
        Number of standard deviations.
    ddof : int
        Delta Degrees of Freedom for standard deviation.
    offset, fillna, use_talib : as usual.

    Returns
    -------
    np.ndarray
        Z‑score values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    mean = sma_ind(
        close, 
        length=length, 
        offset=0, 
        fillna=None, 
        use_talib=use_talib
    )
    std = stdev_ind(
        close, 
        length=length, 
        ddof=ddof, 
        offset=0, 
        fillna=None, 
        use_talib=use_talib
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (close - mean) / (multiplier * std)
    return _apply_offset_fillna(zscore, offset, fillna)


def zscore_ind(
    close: np.ndarray | pl.Series,
    length: int = 30,
    multiplier: float = 1.0,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal rolling Z‑Score (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return zscore_numpy(close, length, multiplier, ddof, offset, fillna, use_talib)


def zscore_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 30,
    multiplier: float = 1.0,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add Z‑Score column to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Name of the column with close prices.
    length : int
        Window length.
    multiplier : float
        Number of standard deviations.
    ddof : int
        Delta Degrees of Freedom for standard deviation.
    offset, fillna, use_talib : as usual.
    output_col : str, optional
        Output column name (default f"ZS_{length}").

    Returns
    -------
    pl.DataFrame
        Original DataFrame with a new column containing the Z‑Score.
    """
    close = df[close_col].to_numpy()
    result = zscore_ind(close, length, multiplier, ddof, offset, fillna, use_talib)
    out_name = output_col or f"ZS_{length}"
    return df.with_columns([pl.Series(out_name, result)])