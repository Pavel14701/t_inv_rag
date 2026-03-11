# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from .. import talib, talib_available
from ..overlap.sma import sma_ind
from ..utils import _apply_offset_fillna


def cdl_doji_numpy(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 10,
    factor: float = 10.0,
    scalar: float = 100.0,
    asint: bool = True,
    naive: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Numpy‑based Doji detection with optional TA‑Lib CDLDOJI.
    """
    # Ensure contiguous float64 arrays
    open_ = np.asarray(open_, dtype=np.float64, copy=False)
    high = np.asarray(high, dtype=np.float64, copy=False)
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    # --- TA-Lib CDLDOJI branch ---
    if use_talib and talib_available and not naive:
        talib_result = talib.CDLDOJI(open_, high, low, close)
        # TA-Lib returns -100, 0, 100 → normalize to your format
        if asint:
            talib_result = (talib_result != 0).astype(np.int32) * scalar
        else:
            talib_result = (talib_result != 0)
        return _apply_offset_fillna(talib_result, offset, fillna)
    # --------------------------------
    # Ensure contiguous arrays (after TA-Lib branch)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # Body absolute value
    body = np.abs(close - open_)
    # High‑low range
    hl_range = high - low
    # Determine reference range
    if naive:
        ref_range = hl_range
    else:
        ref_range = sma_ind(
            hl_range,
            length=length,
            offset=0,
            fillna=None,
            use_talib=use_talib,
        )
    # Doji condition: body < (factor / 100) * ref_range
    doji = body < (factor / 100.0) * ref_range
    # Output formatting
    if asint:
        doji = scalar * doji.astype(np.int32)
    else:
        doji = doji.astype(np.bool_)
    # Apply offset and fillna
    return _apply_offset_fillna(doji, offset, fillna)


def cdl_doji_pat(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 10,
    factor: float = 10.0,
    scalar: float = 100.0,
    asint: bool = True,
    naive: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal Doji detection (accepts numpy arrays or Polars Series).
    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return cdl_doji_numpy(
        open_, high, low, close,
        length=length,
        factor=factor,
        scalar=scalar,
        asint=asint,
        naive=naive,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )


def cdl_doji_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    length: int = 10,
    factor: float = 10.0,
    scalar: float = 100.0,
    asint: bool = True,
    naive: bool = False,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str | None = None,
) -> pl.DataFrame:
    """
    Add Doji signal column to Polars DataFrame.
    """
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    result = cdl_doji_numpy(
        open_arr, high_arr, low_arr, close_arr,
        length=length,
        factor=factor,
        scalar=scalar,
        asint=asint,
        naive=naive,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )
    out_name = output_col or f"CDL_DOJI_{length}_{factor / 100:.2f}"
    return df.with_columns([pl.Series(out_name, result)])
