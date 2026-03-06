# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

from ..overlap import sma_ind
from ..utils import _apply_offset_fillna
from .avs_base import _avs_base, _compute_len_v, _compute_vpcc, _price_v_rolling


# ----------------------------------------------------------------------
# AVSR (resistance)
# ----------------------------------------------------------------------
def avsr_numpy(
    high: np.ndarray,
    low: np.ndarray,       # not used for resistance, kept for symmetry
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
    Adaptive Volume Resistance Level (AVSR) – Numpy version.
    Returns resistance line as numpy array.
    """
    high = np.asarray(high, dtype=np.float64, copy=False)
    close = np.asarray(close, dtype=np.float64, copy=False)
    volume = np.asarray(volume, dtype=np.float64, copy=False)
    for arr in (high, close, volume):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
    vpc, vpr, vm, vpci, deviation_raw = _avs_base(
        close, volume, fast, slow, stand_div, use_talib
    )
    if max_deviation is not None:
        deviation = np.clip(deviation_raw, -max_deviation, max_deviation)
    else:
        deviation = deviation_raw
    lenV = _compute_len_v(vpc, vpci)
    VPCc = _compute_vpcc(vpc)
    price_v = _price_v_rolling(high, vpr, lenV, VPCc)
    adjusted = high + price_v - deviation
    result = sma_ind(adjusted, slow, use_talib=use_talib)
    return _apply_offset_fillna(result, offset, fillna)


def avsr_ind(
    high: np.ndarray | pl.Series,
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
    Universal AVSR (accepts numpy arrays or Polars Series).
    """
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    if isinstance(volume, pl.Series):
        volume = volume.to_numpy()
    low = np.zeros_like(high)  # not used
    return avsr_numpy(
        high, low, close, volume,
        fast, slow, stand_div, max_deviation, offset, fillna, use_talib
    )


def avsr_polars(
    df: pl.DataFrame,
    fast: int,
    slow: int,
    high_col: str = "high",
    close_col: str = "low",
    volume_col: str = "volume",
    date_col: str = "date",
    stand_div: float = 1.0,
    max_deviation: float | None = None,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    output_col: str = "avsr",
) -> pl.DataFrame:
    """
    Add AVSR column to Polars DataFrame.
    """
    high = df[high_col].to_numpy()
    close = df[close_col].to_numpy()
    volume = df[volume_col].to_numpy()
    result = avsr_numpy(
        high, np.zeros_like(high), close, volume,
        fast, slow, stand_div, max_deviation, offset, fillna, use_talib
    )
    return pl.DataFrame({
        date_col: df[date_col],
        output_col: result
    })