# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np
import polars as pl

from ..utils import _apply_offset_fillna
from . import (
    dema_ind,
    ema_ind,
    fwma_ind,
    hma_ind,
    kama_ind,
    linreg_ind,
    midpoint_ind,
    pwma_ind,
    rma_ind,
    sinwma_ind,
    sma_ind,
    ssf_ind,
    swma_ind,
    t3_ind,
    tema_ind,
    trima_ind,
    vidya_ind,
    wma_ind,
)

_MA_FUNCS: dict[str, Callable] = {
    "dema": dema_ind,
    "ema": ema_ind,
    "fwma": fwma_ind,
    "hma": hma_ind,
    "kama": kama_ind,
    "linreg": linreg_ind,
    "midpoint": midpoint_ind,
    "pwma": pwma_ind,
    "rma": rma_ind,
    "sinwma": sinwma_ind,
    "sma": sma_ind,
    "ssf": ssf_ind,
    "swma": swma_ind,
    "t3": t3_ind,
    "tema": tema_ind,
    "trima": trima_ind,
    "vidya": vidya_ind,
    "wma": wma_ind,
}


def zlma(
    close: np.ndarray | pl.Series,
    length: int = 10,
    mamode: str = "ema",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Zero Lag Moving Average (ZLMA).

    Calculated as:
    1. lag = int(0.5 * (length - 1))
    2. close_detrend = 2 * close - shift(close, lag)
    3. Applies the specified moving average (mamode) to close_detrend.

    Parameters
    ---------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Period.
    mamode : str
        Moving average type (e.g., 'ema', 'sma', 'wma', ...).
    offset : int
        Shift the result (positive – forward).
    fillna : float, optional
        Value to fill NaN after the shift.
    use_talib : bool
        If True and TA-Lib is available, use it for the MA (where possible).

    Returns
    -------
    np.ndarray
        ZLMA values.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # Detrending
    lag = int(0.5 * (length - 1))
    if lag > 0:
        close_shifted = np.roll(close, lag)
        close_shifted[:lag] = np.nan
        close_detrend = 2.0 * close - close_shifted
    else:
        close_detrend = close
    ma_func = _MA_FUNCS.get(mamode.lower())
    if ma_func is None:
        raise ValueError(f"Unsupported type of MA: {mamode}")
    # Calling MA with use_talib passed and the rest of the parameters set to default.
    # All MA functions are assumed to have the signature:
    # func(close, length, offset=0, fillna=None, use_talib=True, ...)
    # If any function requires
    # additional arguments (e.g., drift for vidya),
    # they are passed here with default values.
    # In the current implementation, all MA functions in the 
    # project already have these parameters.
    result = ma_func(
        close_detrend,
        length=length,
        offset=0,  # offset will be applied later via _apply_offset_fillna
        fillna=None,
        use_talib=use_talib,
    )
    # Final processing of shift and fillna
    return _apply_offset_fillna(result, offset, fillna)


def zlma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int = 10,
    mamode: str = "ema",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add the ZLMA column to the Polars DataFrame.

    Parameters
    ---------
    df : pl.DataFrame
        Source data.
    close_col : str
        Name of the column with closing prices.
    length, mamode, offset, fillna, use_talib : See zlma.
    output_col : str, optional
        Name of the output column (default: f"ZL_{mamode.upper()}_{length}").

    Returns
    -------
    pl.DataFrame
        The original DataFrame with the added column.
    """
    close = df[close_col].to_numpy()
    result = zlma(close, length, mamode, offset, fillna, use_talib)
    out_name = output_col or f"ZL_{mamode.upper()}_{length}"
    return df.with_columns([pl.Series(out_name, result)])