# -*- coding: utf-8 -*-
import numpy as np
import polars as pl
from numba import jit

from ..utils import _apply_offset_fillna


# ----------------------------------------------------------------------
# Core HWMA loop in Numba
# ----------------------------------------------------------------------
@jit(nopython=True, fastmath=True, cache=True)
def _hwma_numba_core(
    close: np.ndarray,
    na: float,
    nb: float,
    nc: float
) -> np.ndarray:
    """
    Holt-Winter Moving Average core loop.

    Parameters
    ----------
    close : np.ndarray
        1D float64 array of prices.
    na, nb, nc : float
        Smoothing parameters (0 < parameter < 1).

    Returns
    -------
    np.ndarray
        HWMA values (same length as close).
    """
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    last_a = 0.0
    last_v = 0.0
    last_f = close[0]
    for i in range(n):
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * close[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        out[i] = F + V + 0.5 * A
        last_a, last_f, last_v = A, F, V
    return out


# ----------------------------------------------------------------------
# HWMA using Numba (with offset and fillna)
# ----------------------------------------------------------------------
def hwma_numba(
    close: np.ndarray,
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Holt-Winter Moving Average using Numba.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    na, nb, nc : float
        Smoothing parameters (must be between 0 and 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.

    Returns
    -------
    np.ndarray
        HWMA values.
    """
    # Validate parameters
    if not (0 < na < 1):
        na = 0.2
    if not (0 < nb < 1):
        nb = 0.1
    if not (0 < nc < 1):
        nc = 0.1
    close = np.asarray(close, dtype=np.float64, copy=False)
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    hwma = _hwma_numba_core(close, na, nb, nc)
    return _apply_offset_fillna(hwma, offset, fillna)


# ----------------------------------------------------------------------
# Universal HWMA function
# ----------------------------------------------------------------------
def hwma_ind(
    close: np.ndarray | pl.Series,
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    offset: int = 0,
    fillna: float | None = None
) -> np.ndarray:
    """
    Universal Holt-Winter Moving Average (always uses Numba).
    """
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return hwma_numba(close, na, nb, nc, offset, fillna)


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def hwma_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    na: float = 0.2,
    nb: float = 0.1,
    nc: float = 0.1,
    offset: int = 0,
    fillna: float | None = None,
    output_col: str | None = None
) -> pl.DataFrame:
    """
    HWMA for Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    close_col : str
        Name of the column with close prices.
    na, nb, nc : float
        Smoothing parameters (0 < param < 1).
    offset : int
        Shift result.
    fillna : float, optional
        Value to fill NaNs.
    output_col : str, optional
        Output column name (default f"HWMA_{na}_{nb}_{nc}").

    Returns
    -------
    pl.DataFrame
        HWMA series.
    """
    close = df[close_col].to_numpy()
    result = hwma_ind(close, na=na, nb=nb, nc=nc, offset=offset, fillna=fillna)
    out_name = output_col or f"HWMA_{na}_{nb}_{nc}"
    return df.with_columns([pl.Series(out_name, result)])