"""Z-score (standard score) for financial time series.

This module provides functions to compute rolling Z-scores of a price series.
The Z-score measures how many standard deviations an observation is from the
mean.  It is often used as a feature in trading models or as a normalisation
technique.

The implementation supports two backends:
- TA-Lib (if available) for high performance.
- Numba with two algorithms: 'online' (one-pass, faster) and 'two_pass'
    (two-pass, more accurate but slower).

All functions accept either numpy arrays or Polars Series/DataFrames and
return numpy arrays (or Polars DataFrames with the new column added).
"""

from typing import Literal, Optional, Union

import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna
from ..overlap.sma import sma_ind
from ..statistics.stdev import stdev_ind


def zscore_numpy(
    close: np.ndarray,
    length: int = 30,
    multiplier: float = 1.0,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    algorithm: Literal['online', 'two_pass'] = 'online',
) -> np.ndarray:
    """Compute rolling Z-score from a numpy array of prices.

    The Z-score is defined as
    (price - rolling_mean) / (multiplier * rolling_std).
    The rolling window is set by `length`.  By default, the rolling mean and
    standard deviation are computed using TA-Lib if available, otherwise
    a Numba implementation is used.

    Parameters
    ----------
    close : np.ndarray
        1D array of close prices (float64).  Must be contiguous for best
        performance.
    length : int, default 30
        Rolling window size.  The first `length-1` elements of the result
        will be NaN (or filled with `fillna` if offset > 0).
    multiplier : float, default 1.0
        Multiplier applied to the standard deviation.  A value of 2.0 gives
        a Z-score normalised to two standard deviations.
    ddof : int, default 1
        Delta Degrees of Freedom used in the standard deviation calculation.
        0 gives population standard deviation, 1 gives sample standard
        deviation (default).
    offset : int, default 0
        Shift applied to the output array.  Positive values shift the
        Z-score forward (later observations move to earlier positions).
        Positions that become empty are filled with `fillna`.
    fillna : float or None, default None
        Value used to fill positions that become NaN due to the offset.
        If None, NaN is used.
    use_talib : bool, default True
        If True and TA-Lib is installed, it will be used for the rolling
        mean and standard deviation.  Otherwise, the Numba implementation
        is used.
    algorithm : {'online', 'two_pass'}, default 'online'
        Which Numba algorithm to use for standard deviation.  This parameter
        is only effective when `use_talib=False` or TA-Lib is not available.
        - 'online' : one-pass algorithm (fast, may have small numerical
            errors for large windows).
        - 'two_pass' : two-pass algorithm (slower, but more accurate).

    Returns
    -------
    np.ndarray
        Float64 array of Z-score values, same length as the input.
        The first `length-1` values are NaN (or filled if offset is used).

    Raises
    ------
    ValueError
        If `length` < 1.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    >>> zscore_numpy(prices, length=3, multiplier=1.0, ddof=1)
    array([       nan,        nan, -1.       , -1.       , -1.       ,
        -1.       , -1.       , -1.       , -1.       , -1.       ])

    """
    close = np.asarray(close, dtype=np.float64, copy=False)
    if length < 1:
        raise ValueError('length must be >= 1')
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    mean = sma_ind(
        close,
        length=length,
        offset=0,
        fillna=None,
        use_talib=use_talib,
    )
    std = stdev_ind(
        close,
        length=length,
        ddof=ddof,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        algorithm=algorithm,
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        zscore = (close - mean) / (multiplier * std)

    return _apply_offset_fillna(zscore, offset, fillna)


def zscore_ind(
    close: Union[np.ndarray, pl.Series],
    length: int = 30,
    multiplier: float = 1.0,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    algorithm: Literal['online', 'two_pass'] = 'online',
) -> np.ndarray:
    """Universal rolling Z-score that accepts numpy arrays or Polars Series.

    This is a wrapper around :func:`zscore_numpy` that automatically converts
    Polars Series to numpy arrays before processing.  All other parameters
    behave exactly as in :func:`zscore_numpy`.

    Parameters
    ----------
    close : np.ndarray or pl.Series
        1D array or Polars Series of close prices.
    length : int, default 30
        Rolling window size.
    multiplier : float, default 1.0
        Multiplier for standard deviation.
    ddof : int, default 1
        Delta Degrees of Freedom.
    offset : int, default 0
        Shift applied to the output.
    fillna : float or None, default None
        Value to fill positions that become NaN due to offset.
    use_talib : bool, default True
        Use TA-Lib if available.
    algorithm : {'online', 'two_pass'}, default 'online'
        Numba algorithm for standard deviation (ignored if TA-Lib is used).

    Returns
    -------
    np.ndarray
        Float64 array of Z-score values.

    Examples
    --------
    >>> import polars as pl
    >>> s = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    >>> zscore_ind(s, length=3, multiplier=1.0, ddof=1)
    array([       nan,        nan, -1.       , -1.       , -1.       ,
           -1.       , -1.       , -1.       , -1.       , -1.       ])

    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return zscore_numpy(
        close,
        length,
        multiplier,
        ddof,
        offset,
        fillna,
        use_talib,
        algorithm,
    )


def zscore_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int = 30,
    multiplier: float = 1.0,
    ddof: int = 1,
    offset: int = 0,
    fillna: Optional[float] = None,
    use_talib: bool = True,
    algorithm: Literal['online', 'two_pass'] = 'online',
    output_col: Optional[str] = None,
) -> pl.DataFrame:
    """Add a Z-Score column to a Polars DataFrame.

    This function computes the rolling Z-score of a specified column in a
    Polars DataFrame and returns a new DataFrame with the Z-score column
    appended.  The original DataFrame is not modified.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str, default 'close'
        Name of the column containing the price data.
    length : int, default 30
        Rolling window size.
    multiplier : float, default 1.0
        Multiplier for standard deviation.
    ddof : int, default 1
        Delta Degrees of Freedom.
    offset : int, default 0
        Shift applied to the Z-score column.
    fillna : float or None, default None
        Value to fill shifted positions.
    use_talib : bool, default True
        Use TA-Lib if available.
    algorithm : {'online', 'two_pass'}, default 'online'
        Numba algorithm for standard deviation (ignored if TA-Lib is used).
    output_col : str or None, default None
        Name of the output column.  If None, the column will be named
        f'ZS_{length}'.

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the Z-score column appended.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> zscore_polars(df, length=3, output_col='ZSCORE')
    shape: (5, 2)
    ┌───────┬────────┐
    │ close ┆ ZSCORE │
    │ ---   ┆ ---    │
    │ f64   ┆ f64    │
    ╞═══════╪════════╡
    │ 1.0   ┆ NaN    │
    │ 2.0   ┆ NaN    │
    │ 3.0   ┆ -1.0   │
    │ 4.0   ┆ -1.0   │
    │ 5.0   ┆ -1.0   │
    └───────┴────────┘

    """
    close = df[close_col].to_numpy()
    result = zscore_ind(
        close,
        length,
        multiplier,
        ddof,
        offset,
        fillna,
        use_talib,
        algorithm,
    )
    out_name = output_col or f'ZS_{length}'
    return df.with_columns(pl.Series(out_name, result))
