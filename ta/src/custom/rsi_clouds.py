# -*- coding: utf-8 -*-
from typing import cast

import numpy as np
import polars as pl

from ..ma import ma_mode
from ..momentum import rsi_ind
from ..utils import _apply_offset_fillna, _handle_nan_policy


# ----------------------------------------------------------------------
# Core RSI Clouds calculation (NumPy)
# ----------------------------------------------------------------------
def rsi_clouds_numpy(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    rsi_length: int = 14,
    rsi_scalar: float = 100.0,
    rsi_drift: int = 1,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_mamode: str = "ema",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy-based RSI Clouds calculation.

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        OHLC цены.
    rsi_length : int
        Период RSI по средней цене.
    rsi_scalar : float
        Масштаб RSI (обычно 100).
    rsi_drift : int
        Drift для RSI.
    macd_fast, macd_slow, macd_signal : int
        Параметры MACD.
    macd_mamode : str
        Тип сглаживания для MACD (обычно 'ema').
    offset : int
        Сдвиг результата.
    fillna : float, optional
        Чем заполнить NaN.
    use_talib : bool
        Использовать ли TA-Lib в rsi_ind/ma_mode (если доступен).
    nan_policy : str
        'raise', 'ffill', 'bfill', 'both'.
    trim : bool
        Если True — вернуть только валидную часть (без начальных NaN).

    Returns
    -------
    (rsi, macd_line, macd_signal_line, macd_hist)
    """
    # ---- Input to float64 ----
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    # ---- Inf check ----
    for name, arr in (("open", open_), ("high", high), ("low", low), ("close", close)):
        if np.isinf(arr).any():
            raise ValueError(f"Input {name} contains non-finite values (inf or -inf).")

    # ---- NaN handling ----
    open_ = _handle_nan_policy(open_, nan_policy, "open")
    high = _handle_nan_policy(high, nan_policy, "high")
    low = _handle_nan_policy(low, nan_policy, "low")
    close = _handle_nan_policy(close, nan_policy, "close")

    # ---- Length check ----
    n = len(close)
    if not (len(open_) == len(high) == len(low) == n):
        raise ValueError("OHLC arrays must have the same length.")
    if rsi_length < 1 or macd_fast < 1 or macd_slow < 1 or macd_signal < 1:
        raise ValueError("All periods must be >= 1.")

    # ---- Average price ----
    avg = (open_ + high + low + close) / 4.0

    # ---- RSI on avg price ----
    rsi = rsi_ind(
        avg,
        length=rsi_length,
        scalar=rsi_scalar,
        drift=rsi_drift,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        nan_policy=nan_policy,
    )

    # ---- MACD on RSI ----
    # MACD line = MA_fast(RSI) - MA_slow(RSI)
    rsi_fast = cast(np.ndarray, ma_mode(
        macd_mamode,
        rsi,
        length=macd_fast,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        nan_policy=nan_policy,
    ))
    rsi_slow = cast(np.ndarray, ma_mode(
        macd_mamode,
        rsi,
        length=macd_slow,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        nan_policy=nan_policy,
    ))
    macd_line = rsi_fast - rsi_slow

    macd_signal_line = cast(np.ndarray, ma_mode(
        macd_mamode,
        macd_line,
        length=macd_signal,
        offset=0,
        fillna=None,
        use_talib=use_talib,
        nan_policy=nan_policy,
    ))
    macd_hist = macd_line - macd_signal_line

    # ---- Trim (убрать начальные NaN, где MACD ещё не определён) ----
    if trim:
        # Берём первую позицию, где есть валидный macd_signal_line
        valid_idx = np.where(~np.isnan(macd_signal_line))[0]
        if len(valid_idx) > 0:
            start = valid_idx[0]
            rsi = rsi[start:]
            macd_line = macd_line[start:]
            macd_signal_line = macd_signal_line[start:]
            macd_hist = macd_hist[start:]
        else:
            rsi = np.array([])
            macd_line = np.array([])
            macd_signal_line = np.array([])
            macd_hist = np.array([])

    # ---- Offset + fillna ----
    rsi = _apply_offset_fillna(rsi, offset, fillna)
    macd_line = _apply_offset_fillna(macd_line, offset, fillna)
    macd_signal_line = _apply_offset_fillna(macd_signal_line, offset, fillna)
    macd_hist = _apply_offset_fillna(macd_hist, offset, fillna)

    return rsi, macd_line, macd_signal_line, macd_hist


# ----------------------------------------------------------------------
# Universal RSI Clouds (NumPy / Polars)
# ----------------------------------------------------------------------
def rsi_clouds_ind(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    rsi_length: int = 14,
    rsi_scalar: float = 100.0,
    rsi_drift: int = 1,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_mamode: str = "ema",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Universal RSI Clouds (accepts numpy arrays or Polars Series).
    Returns (rsi, macd_line, macd_signal_line, macd_hist).
    """
    if isinstance(open_, pl.Series):
        open_ = open_.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return rsi_clouds_numpy(
        open_,
        high,
        low,
        close,
        rsi_length=rsi_length,
        rsi_scalar=rsi_scalar,
        rsi_drift=rsi_drift,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        macd_mamode=macd_mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=trim,
    )


# ----------------------------------------------------------------------
# Polars integration
# ----------------------------------------------------------------------
def rsi_clouds_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    rsi_length: int = 14,
    rsi_scalar: float = 100.0,
    rsi_drift: int = 1,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_mamode: str = "ema",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    suffix: str = "",
) -> pl.DataFrame:
    """
    RSI Clouds for Polars DataFrame.

    Adds columns:
        RSI_CLOUDS_RSI{suffix}
        RSI_CLOUDS_MACD{suffix}
        RSI_CLOUDS_SIGNAL{suffix}
        RSI_CLOUDS_HIST{suffix}
    """
    open_ = df[open_col].to_numpy()
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    close = df[close_col].to_numpy()

    rsi, macd_line, macd_signal_line, macd_hist = rsi_clouds_numpy(
        open_,
        high,
        low,
        close,
        rsi_length=rsi_length,
        rsi_scalar=rsi_scalar,
        rsi_drift=rsi_drift,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        macd_mamode=macd_mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=False,  # Polars всегда возвращает полную длину
    )

    if not suffix:
        suffix = f"_{rsi_length}_{macd_fast}_{macd_slow}_{macd_signal}"

    return df.with_columns([
        pl.Series(f"RSI_CLOUDS_RSI{suffix}", rsi),
        pl.Series(f"RSI_CLOUDS_MACD{suffix}", macd_line),
        pl.Series(f"RSI_CLOUDS_SIGNAL{suffix}", macd_signal_line),
        pl.Series(f"RSI_CLOUDS_HIST{suffix}", macd_hist),
    ])