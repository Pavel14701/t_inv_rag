# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..overlap import sma_ind
from ..utils import _apply_offset_fillna
from .avs_base import _avs_base, _compute_len_v, _compute_vpcc, _price_v_rolling


# ----------------------------------------------------------------------
# AVSL (support)
# ----------------------------------------------------------------------
def avsl_numpy(
    high: np.ndarray,      # not used for support, kept for symmetry
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    fast: int,
    slow: int,
    stand_div: float = 1.0,
    max_deviation: float | None = None,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Adaptive Volume Support Level (AVSL) – Numpy version.
    Returns support line as numpy array.
    """
    # Ensure contiguous
    low = np.asarray(low, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    volume = np.asarray(volume, dtype=np.float64, copy=False)
    for arr in (low, close, volume):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    # Common base
    vpc, vpr, vm, vpci, deviation_raw = _avs_base(
        close, volume, fast, slow, stand_div, use_talib
    )
    # Apply optional deviation cap
    if max_deviation is not None:
        deviation = np.clip(deviation_raw, -max_deviation, max_deviation)
    else:
        deviation = deviation_raw
    # Dynamic parameters
    lenV = _compute_len_v(vpc, vpci)
    VPCc = _compute_vpcc(vpc)
    # Volume‑adjusted factor
    price_v = _price_v_rolling(low, vpr, lenV, VPCc)
    # Adjusted series (support formula)
    adjusted = low - price_v + deviation
    # Final smoothing
    result = sma_ind(adjusted, slow, use_talib=use_talib)
    return _apply_offset_fillna(result, offset, fillna)


def avsl_ind(
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    volume: np.ndarray | pl.Series,
    fast: int,
    slow: int,
    stand_div: float = 1.0,
    max_deviation: float | None = None,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
) -> np.ndarray:
    """
    Universal AVSL (accepts numpy arrays or Polars Series).
    """
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if isinstance(volume, pl.Series):
        volume = volume.to_numpy()
    # high is not needed for AVSL, pass a dummy array of same length
    high = np.zeros_like(low)
    return avsl_numpy(
        high, low, close, volume,
        fast, slow, stand_div, max_deviation, offset, fillna, use_talib
    )


def avsl_polars(
    df: pl.DataFrame,
    low_col: str,
    close_col: str,
    volume_col: str,
    fast: int,
    slow: int,
    stand_div: float = 1.0,
    max_deviation: float | None = None,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str = "avsl",
) -> pl.DataFrame:
    """
    Add AVSL column to Polars DataFrame.
    """
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()
    volume = df[volume_col].to_numpy()
    result = avsl_numpy(
        np.zeros_like(low), low, close, volume,
        fast, slow, stand_div, max_deviation, offset, fillna, use_talib
    )
    return df.with_columns([pl.Series(output_col, result)])