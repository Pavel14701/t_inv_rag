# -*- coding: utf-8 -*-
import inspect

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna, replace_inf_with_nan
from ..overlap.dema import dema_ind
from ..overlap.ema import ema_ind
from ..overlap.fwma import fwma_ind
from ..overlap.hma import hma_ind
from ..overlap.kama import kama_ind
from ..overlap.linreg import linreg_ind
from ..overlap.midpoint import midpoint_ind
from ..overlap.pwma import pwma_ind
from ..overlap.rma import rma_ind
from ..overlap.sinwma import sinwma_ind
from ..overlap.sma import sma_ind
from ..overlap.ssf import ssf_ind
from ..overlap.swma import swma_ind
from ..overlap.t3 import t3_ind
from ..overlap.tema import tema_ind
from ..overlap.trima import trima_ind
from ..overlap.vidya import vidya_ind
from ..overlap.wma import wma_ind


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


def _call_ma(
    ma_func: Callable,
    arr: np.ndarray,
    length: int,
    use_talib: bool,
) -> np.ndarray:
    """Call a moving-average function with only the kwargs it supports.

    Different MA backends in this package expose different signatures:
    some accept ``use_talib``, some accept ``nan_policy``, some accept
    neither. The detrended series built by ``zlma_ind`` always contains
    NaN values in its first ``lag`` positions (by design), so
    ``nan_policy='ignore'`` is passed to every backend that supports it;
    backends without the parameter propagate NaN naturally.

    Parameters
    ----------
    ma_func : Callable
        One of the ``*_ind`` functions registered in ``_MA_FUNCS``.
    arr : np.ndarray
        Input series (detrended close prices).
    length : int
        MA period.
    use_talib : bool
        Backend flag; forwarded only if ``ma_func`` accepts it.

    Returns
    -------
    np.ndarray
        MA values of the same length as ``arr``.

    """
    params = inspect.signature(ma_func).parameters
    kwargs: dict[str, Any] = {}
    if "use_talib" in params:
        kwargs["use_talib"] = use_talib
    if "nan_policy" in params:
        kwargs["nan_policy"] = "ignore"
    return ma_func(arr, length=length, offset=0, fillna=None, **kwargs)


def zlma_ind(
    close: np.ndarray | pl.Series,
    length: int = 10,
    mamode: str = "ema",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """Zero Lag Moving Average (ZLMA).

    Calculated as:
    1. lag = int(0.5 * (length - 1))
    2. close_detrend = 2 * close - shift(close, lag)
    3. Applies the specified moving average (mamode) to close_detrend.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int
        Period.
    mamode : str
        Moving average type (e.g., 'ema', 'sma', 'wma', ...).
    offset : int
        Shift the result (positive - forward).
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
    # IEEE 754: replace +/-Inf with NaN before detrending so that the
    # error cannot silently turn into +/-Inf (2*Inf - Inf = NaN) inside
    # the composite backends.
    close = close.copy()
    replace_inf_with_nan(close)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # Detrending
    lag = int(0.5 * (length - 1))
    if lag > 0:
        close_shifted = np.roll(close, lag)
        close_shifted[:lag] = np.nan
        close_detrend = 2.0 * close - close_shifted
        # The first `lag` positions of the detrended series are undefined
        # (NaN by construction). Recursive MA backends (ema, dema, kama,
        # sma, ssf, ...) treat NaN inside their seed window as a poison that
        # nullifies the entire output, and the cumulative-sum SMA backend
        # propagates leading NaNs forever. Backfill the undefined head with
        # the first valid detrended value (standard warm-up approximation,
        # same idea as the SMA seed used to initialise EMA). NaNs that come
        # from the actual input data are NOT touched (IEEE 754 propagation).
        valid_mask = ~np.isnan(close_detrend)
        if valid_mask.any():
            first_valid = int(np.argmax(valid_mask))
            close_detrend[:first_valid] = close_detrend[first_valid]
    else:
        close_detrend = close
    ma_func = _MA_FUNCS.get(mamode.lower())
    if ma_func is None:
        raise ValueError(f"Unsupported type of MA: {mamode}")
    # Call the MA with only the keyword arguments it supports:
    # `use_talib` is forwarded to backends that accept it, and
    # `nan_policy='ignore'` is passed to backends that support it because
    # the detrended series intentionally starts with `lag` NaN values.
    result = _call_ma(ma_func, close_detrend, length, use_talib)
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
    """Add the ZLMA column to the Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Source data.
    close_col : str
        Name of the column with closing prices.
    length, mamode, offset, fillna, use_talib : See zlma.
    output_col : str, optional
        Name of the output column (default: f"ZL_{mamode.upper()}_{length}").

    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    length : int, optional
        See the module guide; default mirrors the numpy path.
    mamode : str, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.
    use_talib : bool, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    pl.DataFrame
        The original DataFrame with the added column.

    """
    close = df[close_col].to_numpy()
    result = zlma_ind(close, length, mamode, offset, fillna, use_talib)
    out_name = output_col or f"ZL_{mamode.upper()}_{length}"
    return df.with_columns([pl.Series(out_name, result)])
