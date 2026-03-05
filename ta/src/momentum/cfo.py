# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..overlap import linreg_ind
from ..utils import _apply_offset_fillna


def cfo_numpy(
    close: np.ndarray,
    length: int = 9,
    scalar: float = 100.0,
    drift: int = 1,  # kept for signature compatibility, not used
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based Chande Forecast Oscillator.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    length : int
        Period for linear regression.
    scalar : float
        Multiplier.
    drift : int
        Ignored (only for compatibility).
    offset, fillna, use_talib : as usual.

    Returns
    -------
    np.ndarray
        CFO values.
    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # Time Series Forecast (tsf mode)
    tsf = linreg_ind(
        close, length=length, mode='tsf', offset=0, fillna=None, use_talib=use_talib
    )
    # CFO formula: scalar * (close - tsf) / close
    with np.errstate(divide='ignore', invalid='ignore'):
        cfo = scalar * (close - tsf) / close
    return _apply_offset_fillna(cfo, offset, fillna)


def cfo_ind(
    close: np.ndarray | pl.Series,
    length: int = 9,
    scalar: float = 100.0,
    drift: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal CFO (accepts numpy array or Polars Series).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return cfo_numpy(close, length, scalar, drift, offset, fillna, use_talib)


def cfo_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    date_col: str = "date",
    length: int = 9,
    scalar: float = 100.0,
    drift: int = 1,
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
    length, scalar, drift, offset, fillna, use_talib : as above.
    output_col : str, optional
        Output column name (default f"CFO_{length}").

    Returns
    -------
    pl.DataFrame
    """
    close = df[close_col].to_numpy()
    result = cfo_numpy(close, length, scalar, drift, offset, fillna, use_talib)
    out_name = output_col or f"CFO_{length}"
    return pl.DataFrame({
        date_col: df[date_col],
        out_name: result
    })