"""TOS_STDEVALL indicator (Thinkorswim-style standard deviation bands).

This indicator computes a linear regression line and standard deviation
bands around it, similar to the Thinkorswim platform's STDEVALL study.
The bands are calculated using the last N bars (or the entire series) and
are symmetric around the regression line.

The output includes:
- Central regression line (LR)
- Lower bands (L_1, L_2, ...) for each multiplier
- Upper bands (U_1, U_2, ...) for each multiplier

Functions:
    tos_stdevall_numpy: Numpy-based calculation.
    tos_stdevall_ind: Universal wrapper (numpy or Polars Series).
    tos_stdevall_polars: Polars DataFrame wrapper.

All functions return dictionaries with numpy arrays (or Polars DataFrames
with new columns).
"""

import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna


def tos_stdevall_numpy(  # noqa: C901
    close: np.ndarray,
    length: int | None = None,
    stds: list[float] | None = None,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
) -> dict[str, np.ndarray]:
    """Numpy-based calculation of TOS_STDEVALL bands.

    Returns arrays of length `length` (if specified) or full length.
    NaN/inf values in the input propagate to NaN bands.

    Raises
    ------
    ValueError
        If fewer than 2 data points are given, if `length` < 2 or exceeds
        the data length, if `ddof` is outside [0, n), or if any multiplier
        in `stds` is negative/non-finite.

    """
    close = np.asarray(close, dtype=np.float64)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    if not close.flags.writeable:
        close = close.copy()
    original_len = len(close)
    if original_len < 2:
        raise ValueError('Need at least 2 data points')
    if length is not None:
        if length < 2:
            raise ValueError('length must be >= 2')
        if length > original_len:
            raise ValueError('length cannot exceed data length')
        suffix = f'_{length}'
        calc_close = close[-length:]
        n = length
    else:
        suffix = ''
        calc_close = close
        n = original_len
    if stds is None:
        stds = [1.0, 2.0, 3.0]
    else:
        stds = sorted(stds)
        if any(m < 0 or not np.isfinite(m) for m in stds):
            raise ValueError('stds multipliers must be finite and >= 0')
    if ddof < 0 or ddof >= n:
        raise ValueError('ddof must satisfy 0 <= ddof < number of points')
    x = np.arange(n, dtype=np.float64)
    if np.isfinite(calc_close).all():
        coeffs = np.polyfit(x, calc_close, 1)
        lr = np.polyval(coeffs, x)
        stdev = float(np.std(calc_close, ddof=ddof))
    else:
        # NaN/inf in the input make the regression undefined: propagate
        # NaN instead of letting polyfit emit warnings and partial junk.
        lr = np.full(n, np.nan)
        stdev = np.nan
    base_name = f'TOS_STDEVALL{suffix}'
    res = {
        f'{base_name}_LR': lr,
    }
    for m in stds:
        res[f'{base_name}_L_{m}'] = lr - m * stdev
        res[f'{base_name}_U_{m}'] = lr + m * stdev
    for key, arr in res.items():
        res[key] = _apply_offset_fillna(arr, offset, fillna)
    return res


def tos_stdevall_ind(
    close: np.ndarray | pl.Series,
    length: int | None = None,
    stds: list[float] | None = None,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
) -> dict[str, np.ndarray]:
    """Universal TOS_STDEVALL (accepts numpy array or Polars Series)."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return tos_stdevall_numpy(close, length, stds, ddof, offset, fillna)


def tos_stdevall_polars(
    df: pl.DataFrame,
    close_col: str = 'close',
    length: int | None = None,
    stds: list[float] | None = None,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = '',
) -> pl.DataFrame:
    """Add TOS_STDEVALL columns to a Polars DataFrame.

    This function computes the TOS_STDEVALL bands and appends the resulting
    columns to the DataFrame. The original DataFrame is not modified.

    The added columns are:
        - TOS_STDEVALL{_suffix}_LR
        - TOS_STDEVALL{_suffix}_L_{i} for each i in stds
        - TOS_STDEVALL{_suffix}_U_{i} for each i in stds
    where `_suffix` is:
        - `_{length}` if `length` is not None and `suffix` is empty.
        - If `suffix` is provided, it replaces the automatic suffix.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the close price column.
    close_col : str, default "close"
        Name of the column with close prices.
    length : int or None, default None
        Number of recent bars to use for the regression.
    stds : list of float or None, default None
        Standard deviation multipliers (default [1,2,3]).
    ddof : int, default 1
        Delta Degrees of Freedom.
    offset : int, default 0
        Shift applied to all output columns.
    fillna : float or None, default None
        Value to fill NaN after shift.
    suffix : str, default ""
        Custom suffix to append to column names. If provided, it overrides
        the automatic suffix (`_{length}`). For example, setting suffix="_my"
        would produce column names like "TOS_STDEVALL_my_LR".

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the additional TOS_STDEVALL columns.

    """
    close = df[close_col].to_numpy()
    res_dict = tos_stdevall_numpy(
        close, length, stds, ddof, offset=0, fillna=None
    )
    # If length is not None and less than DataFrame height, pad with NaN
    if length is not None and length < len(df):
        pad_len = len(df) - length
        for key, arr in res_dict.items():
            res_dict[key] = np.concatenate([np.full(pad_len, np.nan), arr])
    # Apply custom suffix if provided
    if suffix:
        if length is not None:
            base = f'TOS_STDEVALL_{length}'
        else:
            base = 'TOS_STDEVALL'
        new_dict = {}
        for key, arr in res_dict.items():
            if key.startswith(base):
                new_key = key.replace(base, f'TOS_STDEVALL{suffix}', 1)
            else:
                new_key = key
            new_dict[new_key] = arr
        res_dict = new_dict
    # Apply offset and fillna AFTER padding
    for key, arr in res_dict.items():
        res_dict[key] = _apply_offset_fillna(arr, offset, fillna)
    return df.with_columns([
        pl.Series(name, arr)
        for name, arr in res_dict.items()
    ])
