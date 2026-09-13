# -*- coding: utf-8 -*-
"""True Strength Index (TSI).

Double-smoothed momentum:

    mom = close - close.shift(1)
    num = EMA(EMA(mom,  short), long)
    den = EMA(EMA(|mom|, short), long)
    TSI    = 100 * num / den
    signal = EMA(TSI, signal_length)

Defaults follow the canonical definition (long=25, short=13,
signal=13). TA-Lib has no TSI; the native path always runs.

IEEE 754 notes
--------------
- strict floating-point arithmetic: ``fastmath=False`` everywhere;
- NaN propagates: a momentum touching a NaN close is NaN and poisons
  the double EMA forever after (documented pandas_ta behaviour);
- a zero denominator yields NaN (explicit rule wins over 0/0 and
  x/0 -> +/-Inf).
"""

import numpy as np
import polars as pl

from .._array_ops import _apply_offset_fillna
from ..overlap.ema import ema_ind


def _ema_from_first_valid(x: np.ndarray, length: int) -> np.ndarray:
    """EMA of a series with a structural NaN warm-up prefix.

    Forward-fills the prefix with the first valid value (so the EMA
    recursion starts cleanly) and masks the contaminated prefix
    (``first_valid + length - 1``). All-NaN input stays all-NaN.
    """
    out = np.full(len(x), np.nan, dtype=np.float64)
    first_valid = np.argmax(~np.isnan(x))
    if np.isnan(x[first_valid]):
        return out  # all-NaN input stays all-NaN
    filled = x.copy()
    filled[:first_valid] = x[first_valid]
    out = ema_ind(filled, length=length, use_talib=False, nan_policy="ignore")
    out[: first_valid + length - 1] = np.nan
    return out


def _double_ema(x: np.ndarray, short: int, long: int) -> np.ndarray:
    """EMA(EMA(x, short), long) warm-up friendly:
    each stage is seeded at the first valid value of its input.
    """
    first = _ema_from_first_valid(x, short)
    return _ema_from_first_valid(first, long)


def tsi_numpy(
    close: np.ndarray,
    long: int = 25,
    short: int = 13,
    signal: int = 13,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute TSI and its signal line using NumPy.

    Parameters
    ----------
    close : np.ndarray
        Close prices (float64).
    long : int
        Second (outer) EMA length (>= 1).
    short : int
        First (inner) EMA length (>= 1).
    signal : int
        EMA length of the signal line (>= 1).
    offset, fillna : as usual.

    fillna : float, optional
        See the module guide; default mirrors the numpy path.
    offset : int, optional
        See the module guide; default mirrors the numpy path.

    Returns
    -------
    tuple of np.ndarray
        (tsi, signal_line) in [-100, 100] (numerically).

    Raises
    ------
    ValueError
        If any period < 1.

    """
    if long < 1:
        raise ValueError("long must be >= 1")
    if short < 1:
        raise ValueError("short must be >= 1")
    if signal < 1:
        raise ValueError("signal must be >= 1")
    close = np.asarray(close, dtype=np.float64, copy=False)
    if close.size == 0:
        return np.array([]), np.array([])
    if not close.flags.c_contiguous:
        close = np.ascontiguousarray(close)
    n = len(close)
    mom = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        mom[1:] = close[1:] - close[:-1]
    num = _double_ema(mom, short, long)
    den = _double_ema(np.abs(mom), short, long)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsi = 100.0 * num / den
    # 0/0 and x/0 -> NaN (undefined oscillation around a flat base).
    tsi = np.where(den == 0.0, np.nan, tsi)  # noqa: RUF069 - exact IEEE zero/sign check
    # Signal line: forward-fill the warm-up NaN prefix with the first
    # valid value so the EMA recursion starts cleanly, then mask the
    # standard prefix (same approach as macd_numpy).
    tsi_filled = tsi.copy()
    first_valid = np.argmax(~np.isnan(tsi))
    if not np.isnan(tsi[first_valid]):
        tsi_filled[:first_valid] = tsi[first_valid]
    signalma = ema_ind(
        tsi_filled, length=signal, use_talib=False, nan_policy="ignore"
    )
    # The EMA recursion is seeded at the first valid TSI value
    # (index ``long + short - 2``), so it is only clean from
    # ``(long + short - 2) + signal - 1``.
    signalma[: long + short + signal - 2] = np.nan
    tsi = _apply_offset_fillna(tsi, offset, fillna)
    signalma = _apply_offset_fillna(signalma, offset, fillna)
    return tsi, signalma


def tsi_ind(
    close: np.ndarray | pl.Series,
    long: int = 25,
    short: int = 13,
    signal: int = 13,
    offset: int = 0,
    fillna: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Universal TSI (accepts numpy array or Polars Series)."""
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return tsi_numpy(
        close,
        long=long,
        short=short,
        signal=signal,
        offset=offset,
        fillna=fillna,
    )


def tsi_polars(
    df: pl.DataFrame,
    close_col: str = "close",
    long: int = 25,
    short: int = 13,
    signal: int = 13,
    offset: int = 0,
    fillna: float | None = None,
    suffix: str = "",
) -> pl.DataFrame:
    """Add TSI and signal columns to a Polars DataFrame.

    Added columns: ``TSI{suffix}`` and ``TSIs{suffix}`` where suffix
    defaults to ``_{long}_{short}``.
    """
    close = df[close_col].cast(pl.Float64).to_numpy()
    tsi, signalma = tsi_numpy(
        close,
        long=long,
        short=short,
        signal=signal,
        offset=offset,
        fillna=fillna,
    )
    if not suffix:
        suffix = f"_{long}_{short}"
    return df.with_columns(
        [
            pl.Series(f"TSI{suffix}", tsi),
            pl.Series(f"TSIs{suffix}", signalma),
        ]
    )
