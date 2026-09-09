"""Z-score candles (cdl_z).

Z-candles transform raw OHLC prices into z-scores based on a rolling window
or global (full-series) statistics. The output is a set of four arrays
(open_Z, high_Z, low_Z, close_Z) that can be used as features.

Functions:
    cdl_z_numpy: Numpy-based calculation (rolling or global).
    cdl_z: Universal wrapper (numpy or Polars Series).
    cdl_z_polars: Polars DataFrame wrapper.

The core logic is implemented in pure numpy and optionally uses the
statistics.zscore_ind function (which itself may use TA-Lib if available).
"""

import numpy as np
import polars as pl

from ..statistics import zscore_ind
from .._array_ops import _apply_offset_fillna


def _safe_z(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return (x - mean) / std, or zeros if std == 0."""
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std


def cdl_z_numpy(  # noqa: C901
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
    """Numpy-based Z-candles calculation.

    Parameters
    ----------
    open_ : np.ndarray
        1D float64 array of open prices.
    high : np.ndarray
        1D float64 array of high prices.
    low : np.ndarray
        1D float64 array of low prices.
    close : np.ndarray
        1D float64 array of close prices.
    length : int, default 30
        Window length for rolling z-score (ignored if full=True).
    full : bool, default False
        If True, use global (full-series) statistics instead of rolling.
    ddof : int, default 1
        Delta degrees of freedom for standard deviation calculation.
    offset : int, default 0
        Shift applied to outputs (positive = forward shift).
    fillna : float or None, default None
        Value to replace NaN after shift.
    use_talib : bool, default True
        Whether to use TA-Lib for rolling z-score (if available).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - 'open_Z{auto_suffix}' etc., where suffix is:
            - '_a' if full=True
            - f'_{length}_{ddof}' if full=False

    """
    # Ensure float64 and contiguous
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not open_.flags.c_contiguous:
        open_ = np.ascontiguousarray(open_)
    if not open_.flags.writeable:
        open_ = open_.copy()
    if not high.flags.c_contiguous:
        high = np.ascontiguousarray(high)
    if not high.flags.writeable:
        high = high.copy()
    if not low.flags.c_contiguous:
        low = np.ascontiguousarray(low)
    if not low.flags.writeable:
        low = low.copy()
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()
    if full:
        mean_o = open_.mean()
        mean_h = high.mean()
        mean_l = low.mean()
        mean_c = close.mean()
        std_o = open_.std(ddof=ddof)
        std_h = high.std(ddof=ddof)
        std_l = low.std(ddof=ddof)
        std_c = close.std(ddof=ddof)
        z_open = _safe_z(open_, mean_o, std_o)
        z_high = _safe_z(high, mean_h, std_h)
        z_low = _safe_z(low, mean_l, std_l)
        z_close = _safe_z(close, mean_c, std_c)
        suffix = 'a'
    else:
        z_open = zscore_ind(
            open_,
            length=length,
            ddof=ddof,
            use_talib=use_talib
        )
        z_high = zscore_ind(
            high,
            length=length,
            ddof=ddof,
            use_talib=use_talib
        )
        z_low = zscore_ind(
            low,
            length=length,
            ddof=ddof,
            use_talib=use_talib
        )
        z_close = zscore_ind(
            close,
            length=length,
            ddof=ddof,
            use_talib=use_talib
        )
        suffix = f'_{length}_{ddof}'
    # Apply offset and fillna
    result = {
        'open_Z': _apply_offset_fillna(z_open, offset, fillna),
        'high_Z': _apply_offset_fillna(z_high, offset, fillna),
        'low_Z': _apply_offset_fillna(z_low, offset, fillna),
        'close_Z': _apply_offset_fillna(z_close, offset, fillna),
    }
    # Rename keys with suffix
    if suffix:
        return {f'{key}{suffix}': arr for key, arr in result.items()}
    return result


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
    """Universal Z-candles (accepts numpy arrays or Polars Series).

    Converts Polars Series to numpy, then calls cdl_z_numpy.

    Parameters
    ----------
    open_ : np.ndarray or pl.Series
        1D float64 array or Polars Series of open prices.
    high : np.ndarray or pl.Series
        1D float64 array or Polars Series of high prices.
    low : np.ndarray or pl.Series
        1D float64 array or Polars Series of low prices.
    close : np.ndarray or pl.Series
        1D float64 array or Polars Series of close prices.
    length : int, default 30
        Window length for rolling z-score (ignored if full=True).
    full : bool, default False
        If True, use global (full-series) statistics instead of rolling.
    ddof : int, default 1
        Delta degrees of freedom for standard deviation calculation.
    offset : int, default 0
        Shift applied to outputs (positive = forward shift).
    fillna : float or None, default None
        Value to replace NaN after shift.
    use_talib : bool, default True
        Whether to use TA-Lib for rolling z-score (if available).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - 'open_Z_{suffix}', 'high_Z_{suffix}', 'low_Z_{suffix}',
        'close_Z_{suffix}' where suffix is '_a' if full=True,
        or '_{length}_{ddof}' otherwise.

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
        open_, high, low, close,
        length=length, full=full, ddof=ddof,
        offset=offset, fillna=fillna, use_talib=use_talib,
    )


def cdl_z_polars(
    df: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    date_col: str = 'date',
    length: int = 30,
    full: bool = False,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    suffix: str = '',
) -> pl.DataFrame:
    """Add Z-candle columns to a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with OHLC columns.
    open_col : str, default 'open'
        Name of the open column.
    high_col : str, default 'high'
        Name of the high column.
    low_col : str, default 'low'
        Name of the low column.
    close_col : str, default 'close'
        Name of the close column.
    date_col : str, default 'date'
        Name of the date/time column (included in output).
    length : int, default 30
        Window length for rolling z-score (ignored if full=True).
    full : bool, default False
        If True, use global (full-series) statistics instead of rolling.
    ddof : int, default 1
        Delta degrees of freedom for standard deviation calculation.
    offset : int, default 0
        Shift applied to outputs (positive = forward shift).
    fillna : float or None, default None
        Value to replace NaN after shift.
    use_talib : bool, default True
        Whether to use TA-Lib for rolling z-score (if available).
    suffix : str, default ""
        Custom suffix for column names. If provided, it overrides the
        auto-generated suffix (`_a` for full, or `_{length}_{ddof}`).

    Returns
    -------
    pl.DataFrame
        New DataFrame with date column and four Z-score columns:
        open_Z{suffix}, high_Z{suffix}, low_Z{suffix}, close_Z{suffix}.

    """
    open_arr = df[open_col].to_numpy()
    high_arr = df[high_col].to_numpy()
    low_arr = df[low_col].to_numpy()
    close_arr = df[close_col].to_numpy()
    result = cdl_z_numpy(
        open_arr, high_arr, low_arr, close_arr,
        length=length, full=full, ddof=ddof,
        offset=offset, fillna=fillna, use_talib=use_talib,
    )
    # Apply custom suffix if provided
    if suffix:
        new_dict = {}
        for key, arr in result.items():
            base = key.split('_Z')[0]  # "open", "high", etc.
            new_dict[f'{base}_Z{suffix}'] = arr
        result = new_dict
    # Build output DataFrame
    out_df = pl.DataFrame({date_col: df[date_col]})
    for name, arr in result.items():
        out_df = out_df.with_columns(pl.Series(name, arr))
    return out_df
