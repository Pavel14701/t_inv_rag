# -*- coding: utf-8 -*-
"""Know Sure Thing (KST) -- smoothed multi-period Rate of Change.

    KST = SMA(ROC(roc1), sma1)
        + 2  * SMA(ROC(roc2), sma2)
        + 3  * SMA(ROC(roc3), sma3)
        + 4  * SMA(ROC(roc4), sma4)
    signal = SMA(KST, signal)

Defaults follow the canonical definition (10/15/20/30 ROC lengths with
10/10/10/15 SMA smoothings, signal 9). TA-Lib has no KST; the native
path always runs.

IEEE 754 notes
--------------
- NaN propagates: ROC windows touching a NaN close yield NaN, and any
  NaN inside an SMA window poisons that SMA value only;
- no fastmath, no fabricated values.
"""

import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna
from ..overlap.sma import sma_ind
from .roc import roc_ind


def kst_numpy(
    close: np.ndarray,
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    signal: int = 9,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute KST and its signal line using NumPy.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    roc1..roc4 : int
        ROC lengths (>= 1).
    sma1..sma4 : int
        SMA smoothing lengths for the ROC lines (>= 1).
    signal : int
        SMA length of the signal line (>= 1).
    offset, fillna : as usual.

    Returns
    -------
    tuple of np.ndarray
        (kst, signal_line).

    Raises
    ------
    ValueError
        If any period < 1.

    """
    for name, val in (
        ("roc1", roc1),
        ("roc2", roc2),
        ("roc3", roc3),
        ("roc4", roc4),
        ("sma1", sma1),
        ("sma2", sma2),
        ("sma3", sma3),
        ("sma4", sma4),
        ("signal", signal),
    ):
        if val < 1:
            raise ValueError(f"{name} must be >= 1")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if close.size == 0:
        return np.array([]), np.array([])
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)

    def _smooth_roc(length: int, smoothing: int) -> np.ndarray:
        roc = roc_ind(close, length=length, use_talib=False)
        return sma_ind(
            roc, length=smoothing, use_talib=False, nan_policy="ignore"
        )

    kst = (
        _smooth_roc(roc1, sma1)
        + 2.0 * _smooth_roc(roc2, sma2)
        + 3.0 * _smooth_roc(roc3, sma3)
        + 4.0 * _smooth_roc(roc4, sma4)
    )
    signalma = sma_ind(
        kst, length=signal, use_talib=False, nan_policy="ignore"
    )
    kst = _apply_offset_fillna(kst, offset, fillna)
    signalma = _apply_offset_fillna(signalma, offset, fillna)
    return kst, signalma


def kst_ind(
    close: np.ndarray | pl.Series,
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    signal: int = 9,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal KST (accepts numpy array or Polars Series)."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return kst_numpy(
        close,
        roc1,
        roc2,
        roc3,
        roc4,
        sma1,
        sma2,
        sma3,
        sma4,
        signal,
        offset,
        fillna,
    )


def kst_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    signal: int = 9,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """Add KST and signal columns to a Polars DataFrame.

    Added columns: ``KST{suffix}`` and ``KSTs{suffix}`` where suffix
    defaults to ``_{roc1}_{roc2}_{roc3}_{roc4}``.
    """
    close = df[close_col].cast(pl.Float64).to_numpy()
    kst, signalma = kst_numpy(
        close,
        roc1,
        roc2,
        roc3,
        roc4,
        sma1,
        sma2,
        sma3,
        sma4,
        signal,
        offset,
        fillna,
    )
    if not suffix:
        suffix = f"_{roc1}_{roc2}_{roc3}_{roc4}"
    return df.with_columns(
        [
            pl.Series(f"KST{suffix}", kst),
            pl.Series(f"KSTs{suffix}", signalma),
        ]
    )
