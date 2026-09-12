# -*- coding: utf-8 -*-
"""Indicator pre-computation for the order block pipeline."""

from __future__ import annotations

import numpy as np

from ..._array_ops import _rolling_max_numba, _rolling_min_numba
from ...momentum.macd import macd_ind
from ...momentum.rsi import rsi_ind
from ...overlap.sma import sma_ind
from ...trend.adx import adx_ind
from ...volatility.atr import atr_ind
from .config import OrderBlockConfig


def precompute_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    cfg: OrderBlockConfig,
) -> dict[str, np.ndarray]:
    """Compute every indicator the pipeline needs, once, up front."""
    atr = atr_ind(
        high,
        low,
        close,
        length=cfg.atr_period,
        use_talib=cfg.use_talib,
    )
    avg_volume = sma_ind(volume, cfg.volume_window, use_talib=cfg.use_talib)
    local_highs = _rolling_max_numba(high, cfg.liquidity_window)
    local_lows = _rolling_min_numba(low, cfg.liquidity_window)
    zone_low = close - cfg.zone_atr_multiplier * atr
    zone_high = close + cfg.zone_atr_multiplier * atr
    result = {
        "atr": atr,
        "avg_volume": avg_volume,
        "local_highs": local_highs,
        "local_lows": local_lows,
        "zone_low": zone_low,
        "zone_high": zone_high,
    }
    if cfg.use_adx_filter:
        adx, adxr, di_plus, di_minus = adx_ind(
            high,
            low,
            close,
            length=cfg.adx_period,
            use_talib=cfg.use_talib,
        )
        result["adx"] = adx
        result["di_plus"] = di_plus
        result["di_minus"] = di_minus
    if cfg.use_rsi_confirmation:
        result["rsi"] = rsi_ind(
            close,
            length=cfg.rsi_period,
            use_talib=cfg.use_talib,
        )
    if cfg.use_macd_confirmation:
        macd, signal, hist = macd_ind(
            close,
            fast=cfg.macd_fast,
            slow=cfg.macd_slow,
            signal=cfg.macd_signal,
            use_talib=cfg.use_talib,
        )
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = hist
    return result


def compute_lookback(
    indicators: dict[str, np.ndarray],
    cfg: OrderBlockConfig,
) -> int:
    """Breakout lookback derived from median ATR, clamped to bounds."""
    if not cfg.use_dynamic_lookback:
        return cfg.lookback_min
    median_atr = np.nanmedian(indicators["atr"])
    if not np.isfinite(median_atr):
        return cfg.lookback_min
    lookback = int(
        min(
            cfg.lookback_max,
            max(cfg.lookback_min, median_atr * cfg.lookback_atr_multiplier),
        )
    )
    return max(1, lookback)
