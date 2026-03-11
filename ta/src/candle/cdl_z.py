# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..statistics import zscore_ind
from ..utils import _apply_offset_fillna


def safe_z(x, mean, std):
    return np.full_like(x, 0.0) if std == 0 else (x - mean) / std


def cdl_z_numpy(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 30,
    full: bool = False,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> dict[str, np.ndarray]:
    """
    Numpy‑based Z Candles calculation.

    Returns a dictionary with keys: 'open_Z', 'high_Z', 'low_Z', 'close_Z'.
    """
    # Ensure contiguous
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    for arr in (open_, high, low, close):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    if full:
        mean_o = open_.mean()
        mean_h = high.mean()
        mean_l = low.mean()
        mean_c = close.mean()
        std_o = open_.std(ddof=ddof)
        std_h = high.std(ddof=ddof)
        std_l = low.std(ddof=ddof)
        std_c = close.std(ddof=ddof)
        z_open = safe_z(open_, mean_o, std_o)
        z_high = safe_z(high, mean_h, std_h)
        z_low = safe_z(low, mean_l, std_l)
        z_close = safe_z(close, mean_c, std_c)
    else:
        z_open = zscore_ind(open_, length=length, ddof=ddof, use_talib=use_talib)
        z_high = zscore_ind(high, length=length, ddof=ddof, use_talib=use_talib)
        z_low = zscore_ind(low, length=length, ddof=ddof, use_talib=use_talib)
        z_close = zscore_ind(close, length=length, ddof=ddof, use_talib=use_talib)
    suffix = "a" if full else f"_{length}_{ddof}"
    return {
        f"open_Z{suffix}": _apply_offset_fillna(z_open, offset, fillna),
        f"high_Z{suffix}": _apply_offset_fillna(z_high, offset, fillna),
        f"low_Z{suffix}": _apply_offset_fillna(z_low, offset, fillna),
        f"close_Z{suffix}": _apply_offset_fillna(z_close, offset, fillna),
    }


def cdl_z(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    length: int = 30,
    full: bool = False,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> dict[str, np.ndarray]:
    """
    Universal Z Candles (accepts numpy arrays or Polars Series).
    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return cdl_z_numpy(
        open_, high, low, close, length, full, ddof, offset, fillna, use_talib
    )


def cdl_z_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    date_col: str = "date",
    length: int = 30,
    full: bool = False,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add Z Candle columns to a Polars DataFrame.

    Columns added:
        open_Z{suffix}, high_Z{suffix}, low_Z{suffix}, close_Z{suffix}
    where suffix is either "_a" (if full=True) or f"_{length}_{ddof}".

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    open_col, high_col, low_col, close_col : str
        Column names for OHLC prices.
    date_col : str
        Name of the date/time column (included in output).
    length, full, ddof, offset, fillna, use_talib : as above.
    suffix : str
        Custom suffix (overrides automatic one).

    Returns
    -------
    pl.DataFrame
        New DataFrame with date and the four Z‑score columns.
    """
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    res_dict = cdl_z_numpy(
        open_arr, high_arr, low_arr, close_arr,
        length=length,
        full=full,
        ddof=ddof,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
    )
    # If suffix is provided, replace the auto‑generated part
    if suffix:
        new_dict = {}
        for key, arr in res_dict.items():
            # expected key format: "open_Z{auto}"
            base = key.split("_Z")[0]  # e.g. "open"
            new_dict[f"{base}_Z{suffix}"] = arr
        res_dict = new_dict
    # Build output DataFrame with date column
    out_df = pl.DataFrame({date_col: df[date_col]})
    for name, arr in res_dict.items():
        out_df = out_df.with_columns(pl.Series(name, arr))
    return out_df