# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..utils import _apply_offset_fillna


def tos_stdevall_numpy(
    close: np.ndarray,
    length: int | None = None,
    stds: list[float] | None = None,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Numpy‑based TOS_STDEVALL calculation.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64). If `length` is given, only the last `length`
        points are used.
    length : int, optional
        Number of recent bars to consider. If None, all data are used.
    stds : list of float, optional
        Multipliers for the standard deviation bands (default [1,2,3]).
    ddof : int
        Delta Degrees of Freedom for the standard deviation (default 1).
    offset : int
        Global shift applied to all result columns.
    fillna : float, optional
        Value to fill NaNs after shifting.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with column names as keys and numpy arrays as values.
        Keys are:
            f"TOS_STDEVALL{_suffix}_LR"            – central regression line
            f"TOS_STDEVALL{_suffix}_L_{i}"        – lower band for multiplier i
            f"TOS_STDEVALL{_suffix}_U_{i}"        – upper band for multiplier i
        where `_suffix` = f"_{length}" if length is not None else "".
    """
    close = np.asarray(close, dtype=np.float64)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    # Handle length
    if length is not None:
        length = int(length)
        if length < 2:
            raise ValueError("length must be >= 2")
        close = close[-length:]
        suffix = f"_{length}"
    else:
        length = len(close)
        suffix = ""
    if length < 2:
        raise ValueError("Need at least 2 data points")
    # Default stds
    if stds is None:
        stds = [1.0, 2.0, 3.0]
    else:
        stds = sorted(stds)  # ensure increasing order
    # Linear regression using polyfit (via numpy.polynomial for newer API)
    x = np.arange(length, dtype=np.float64)
    coeffs = np.polyfit(x, close, 1)          # coeffs[0] = slope, coeffs[1] = intercept
    lr = np.polyval(coeffs, x)                # regression line values
    # Standard deviation of close
    stdev = np.std(close, ddof=ddof)
    # Prepare result dictionary
    res = {}
    base_name = f"TOS_STDEVALL{suffix}"
    res[f"{base_name}_LR"] = lr
    for i in stds:
        res[f"{base_name}_L_{i}"] = lr - i * stdev
        res[f"{base_name}_U_{i}"] = lr + i * stdev
    # Apply offset and fillna to every column
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
    """
    Universal TOS_STDEVALL (accepts numpy array or Polars Series).

    Parameters
    ----------
    close : np.ndarray or pl.Series
        Close prices.
    length : int, optional
        Number of recent bars to consider.
    stds : list of float, optional
        Standard deviation multipliers (default [1,2,3]).
    ddof : int
        Delta Degrees of Freedom (default 1).
    offset, fillna : as usual.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of result arrays.
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return tos_stdevall_numpy(close, length, stds, ddof, offset, fillna)


def tos_stdevall_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    length: int | None = None,
    stds: list[float] | None = None,
    ddof: int = 1,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """
    Add TOS_STDEVALL columns to a Polars DataFrame.

    The following columns are added:
        - TOS_STDEVALL{_suffix}_LR
        - TOS_STDEVALL{_suffix}_L_{i}  for each i in stds
        - TOS_STDEVALL{_suffix}_U_{i}  for each i in stds
    where `_suffix` = f"_{length}" if length is not None, or an empty string.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.
    close_col : str
        Name of the column with close prices.
    length : int, optional
        Number of recent bars to use.
    stds : list of float, optional
        Standard deviation multipliers.
    ddof : int
        Delta Degrees of Freedom.
    offset : int
        Global shift applied to all new columns.
    fillna : float, optional
        Value to fill NaNs after shifting.
    suffix : str
        Additional suffix to append to column names (overrides the automatic one).

    Returns
    -------
    pl.DataFrame
        Original DataFrame with new columns added.
    """
    close = df[close_col].to_numpy()
    res_dict = tos_stdevall_numpy(close, length, stds, ddof, offset, fillna)
    # If a custom suffix is provided, replace the automatic one.
    # The automatic suffix is built into the keys. We'll rename the columns.
    if suffix:
        new_dict = {}
        for key, arr in res_dict.items():
            # Replace the base name part with custom suffix
            # e.g. "TOS_STDEVALL_20_LR" -> "TOS_STDEVALL_custom_LR"
            base = "TOS_STDEVALL"
            if length is not None:
                base += f"_{length}"
            if key.startswith(base):
                new_key = key.replace(base, f"TOS_STDEVALL{suffix}", 1)
            else:
                new_key = key  # fallback
            new_dict[new_key] = arr
        res_dict = new_dict
    # Add all series to the DataFrame
    return df.with_columns([pl.Series(name, arr) for name, arr in res_dict.items()])