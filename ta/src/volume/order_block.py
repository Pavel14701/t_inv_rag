# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
from scipy.signal import find_peaks

from ..momentum import macd_ind, rsi_ind
from ..overlap import sma_ind
from ..trend import adx_ind
from ..utils import _rolling_max_numba, _rolling_min_numba

# Our own Numba‑accelerated indicator functions
from ..volatility import atr_ind


@dataclass
class OrderBlock:
    """Represents a confirmed order block."""
    block_type: str          # "supply" or "demand"
    start: datetime       # time of the original peak/valley
    break_: datetime      # time of the breakout candle
    retest: datetime      # time of the retest candle
    zone_low: float          # lower bound of the zone
    zone_high: float         # upper bound of the zone
    strength: float = 0.0    # optional strength score


# ----------------------------------------------------------------------
# ZigZag using SciPy find_peaks
# ----------------------------------------------------------------------
def zigzag_peaks_valleys(
    high: np.ndarray,
    low: np.ndarray,
    prominence_peak: float,
    prominence_valley: float,
    distance: int,
    width: float | None,
    wlen: int | None,
    rel_height: float,
    plateau_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (peak_indices, valley_indices) as numpy arrays.
    """
    # Peaks in high series
    peak_indices, _ = find_peaks(
        x=high,
        prominence=prominence_peak,
        distance=distance,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size,
    )
    # Valleys in low series (invert and find peaks)
    valley_indices, _ = find_peaks(
        x=-low,
        prominence=prominence_valley,
        distance=distance,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size,
    )
    return peak_indices, valley_indices


# ----------------------------------------------------------------------
# Indicator pre‑computation (using our own functions)
# ----------------------------------------------------------------------
def precompute_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr_period: int,
    volume_window: int,
    liquidity_window: int,
    zone_atr_multiplier: float,
    adx_period: int | None,
    rsi_period: int | None,
    macd_fast: int | None,
    macd_slow: int,
    macd_signal: int,
    use_talib: bool = True,
) -> dict:
    """Computes all needed series as numpy arrays."""
    # ATR
    atr = atr_ind(high, low, close, length=atr_period, use_talib=use_talib)
    # Average volume (SMA)
    avg_volume = sma_ind(volume, volume_window, use_talib=use_talib)
    # Local highs/lows (rolling max/min)
    local_highs = _rolling_max_numba(high, liquidity_window)
    local_lows = _rolling_min_numba(low, liquidity_window)
    zone_low = close - zone_atr_multiplier * atr
    zone_high = close + zone_atr_multiplier * atr
    result = {
        "atr": atr,
        "avg_volume": avg_volume,
        "local_highs": local_highs,
        "local_lows": local_lows,
        "zone_low": zone_low,
        "zone_high": zone_high,
    }

    if adx_period is not None:
        adx, adxr, di_plus, di_minus = adx_ind(
            high, low, close, 
            length=adx_period, 
            use_talib=use_talib
        )
        result["adx"] = adx
        result["di_plus"] = di_plus
        result["di_minus"] = di_minus
    if rsi_period is not None:
        result["rsi"] = rsi_ind(close, length=rsi_period, use_talib=use_talib)
    if macd_fast is not None:
        macd, signal, hist = macd_ind(
            close, fast=macd_fast, 
            slow=macd_slow, 
            signal=macd_signal, 
            use_talib=use_talib
        )
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = hist
    return result


# ----------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------
def generate_block_candidates(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    dates: np.ndarray,  # array of datetime objects or ints
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    indicators: dict,
    lookback_min: int,
    lookback_max: int,
    lookback_atr_multiplier: float,
    use_dynamic_lookback: bool,
    liquidity_tolerance: float,
    multiple_breakouts: bool,
    breakout_volume_threshold: float,
    use_adx_filter: bool,
    adx_threshold: float,
) -> list[dict]:
    """
    Returns list of candidate dictionaries with keys:
        block_type, idx, break_idx, strength, start_date, etc.
    """
    n = len(high)
    candidates = []

    # Determine lookback (scalar for simplicity)
    if use_dynamic_lookback:
        median_atr = np.nanmedian(indicators["atr"])
        lookback = int(min(
            lookback_max, max(lookback_min, median_atr * lookback_atr_multiplier)
        ))
    else:
        lookback = lookback_min

    # Helper for breakout condition
    def is_good_breakout(break_idx, extreme_idx, direction):
        # Volume check
        if volume[break_idx] <= breakout_volume_threshold \
            * indicators["avg_volume"][break_idx]:
            return False
        if use_adx_filter:
            adx = indicators["adx"][break_idx]
            di_plus = indicators["di_plus"][break_idx]
            di_minus = indicators["di_minus"][break_idx]
            if np.isnan(adx) or adx < adx_threshold:
                return False
            if direction == "down" and di_minus <= di_plus:
                return False
            if direction == "up" and di_plus <= di_minus:
                return False
        return True

    # Convert indices to sets for fast lookup
    peak_set = set(peak_indices)
    valley_set = set(valley_indices)

    for i in range(lookback, n):
        idx = i - lookback

        # Supply candidate (peak)
        if idx in peak_set:
            # Liquidity check
            if abs(indicators["local_highs"][idx] - high[idx]) >= liquidity_tolerance:
                continue

            if multiple_breakouts:
                max_future = min(i + lookback_max, n)
                for brk in range(i, max_future):
                    if low[brk] < low[idx] and is_good_breakout(brk, idx, "down"):
                        strength = (low[idx] - low[brk]) / low[idx]  # simple move %
                        candidates.append({
                            "block_type": "supply",
                            "idx": idx,
                            "break_idx": brk,
                            "strength": strength,
                            "start_date": dates[idx],
                        })
            else:
                if low[i] < low[idx] and is_good_breakout(i, idx, "down"):
                    candidates.append({
                        "block_type": "supply",
                        "idx": idx,
                        "break_idx": i,
                        "strength": 1.0,
                        "start_date": dates[idx],
                    })

        # Demand candidate (valley)
        elif idx in valley_set:
            if abs(indicators["local_lows"][idx] - low[idx]) >= liquidity_tolerance:
                continue

            if multiple_breakouts:
                max_future = min(i + lookback_max, n)
                for brk in range(i, max_future):
                    if high[brk] > high[idx] and is_good_breakout(brk, idx, "up"):
                        strength = (high[brk] - high[idx]) / high[idx]
                        candidates.append({
                            "block_type": "demand",
                            "idx": idx,
                            "break_idx": brk,
                            "strength": strength,
                            "start_date": dates[idx],
                        })
            else:
                if high[i] > high[idx] and is_good_breakout(i, idx, "up"):
                    candidates.append({
                        "block_type": "demand",
                        "idx": idx,
                        "break_idx": i,
                        "strength": 1.0,
                        "start_date": dates[idx],
                    })

    return candidates


# ----------------------------------------------------------------------
# Validation and retest
# ----------------------------------------------------------------------
def validate_block_candidates(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    dates: np.ndarray,
    candidates: list[dict],
    indicators: dict,
    confirmation_window: int,
    min_reaction_size: float,
    use_rsi_confirmation: bool,
    rsi_overbought: float,
    rsi_oversold: float,
    use_macd_confirmation: bool,
) -> list[OrderBlock]:
    confirmed = []
    for cand in candidates:
        idx = cand["idx"]
        break_idx = cand["break_idx"]
        is_supply = cand["block_type"] == "supply"

        end_idx = min(break_idx + confirmation_window, len(close))
        zone_low = indicators["zone_low"][idx]
        zone_high = indicators["zone_high"][idx]
        avg_vol = indicators["avg_volume"]

        for j in range(break_idx + 1, end_idx):
            # Retest must touch the zone
            if is_supply:
                price_in_zone = zone_low <= low[j] <= zone_high
            else:
                price_in_zone = zone_low <= high[j] <= zone_high
            if not price_in_zone:
                continue

            # Volume confirmation
            if volume[j] <= avg_vol[j]:
                continue

            # Optional RSI
            if use_rsi_confirmation:
                rsi = indicators["rsi"][j]
                if is_supply and (rsi < rsi_overbought):
                    continue
                if not is_supply and (rsi > rsi_oversold):
                    continue

            # Optional MACD
            if use_macd_confirmation:
                hist = indicators["macd_hist"][j]
                if is_supply and hist > 0:
                    continue
                if not is_supply and hist < 0:
                    continue

            # Reaction size
            close_j = close[j]
            if is_supply:
                reaction = (zone_low - close_j) / zone_low
            else:
                reaction = (close_j - zone_high) / zone_high

            if reaction >= min_reaction_size:
                strength = cand.get(
                    "strength", 1.0
                ) * (
                    volume[j] / avg_vol[j]
                ) * reaction
                block = OrderBlock(
                    block_type=cand["block_type"],
                    start=cand["start_date"],
                    break_=dates[break_idx],
                    retest=dates[j],
                    zone_low=zone_low,
                    zone_high=zone_high,
                    strength=strength,
                )
                confirmed.append(block)
                break  # first retest wins
    return confirmed


# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------
def cluster_order_blocks(
    blocks: list[OrderBlock], 
    price_tolerance: float
) -> list[OrderBlock]:
    if not blocks:
        return blocks
    # Sort by mid price
    blocks_sorted = sorted(blocks, key=lambda b: (b.zone_low + b.zone_high) / 2)
    merged = []
    current = blocks_sorted[0]
    for b in blocks_sorted[1:]:
        if b.block_type == current.block_type:
            price_dist = abs((
                b.zone_low + b.zone_high
            ) / 2 - (
                current.zone_low + current.zone_high
            ) / 2)
            if price_dist <= price_tolerance * (
                (current.zone_high + current.zone_low) / 2
            ):
                # merge
                current.start = min(current.start, b.start)
                current.break_ = max(current.break_, b.break_)
                current.retest = max(current.retest, b.retest)
                current.zone_low = min(current.zone_low, b.zone_low)
                current.zone_high = max(current.zone_high, b.zone_high)
                current.strength = max(current.strength, b.strength)
                continue
        merged.append(current)
        current = b
    merged.append(current)
    return merged


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def identify_order_blocks(
    df: pl.DataFrame,
    # ZigZag parameters (passed to scipy.signal.find_peaks)
    zigzag_prominence_peak: float = 0.01,
    zigzag_prominence_valley: float = 0.01,
    zigzag_distance: int = 5,
    zigzag_width: float | None = None,
    zigzag_wlen: int | None = None,
    zigzag_rel_height: float = 0.5,
    zigzag_plateau_size: int | None = None,
    # Breakout & lookback
    lookback_min: int = 5,
    lookback_max: int = 50,
    lookback_atr_multiplier: float = 2.0,
    use_dynamic_lookback: bool = True,
    multiple_breakouts: bool = False,
    breakout_volume_threshold: float = 1.0,
    # Trend filter
    use_adx_filter: bool = False,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    # Additional confirmation
    use_rsi_confirmation: bool = False,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    use_macd_confirmation: bool = False,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    # Retest & reaction
    confirmation_window: int = 10,
    min_reaction_size: float = 0.002,
    # Zone sizing
    atr_period: int = 14,
    zone_atr_multiplier: float = 1.0,
    # Volume & liquidity
    volume_window: int = 20,
    liquidity_window: int = 10,
    liquidity_tolerance: float = 0.001,
    # Clustering
    cluster_blocks: bool = False,
    cluster_price_tolerance: float = 0.001,
    # Historical depth
    max_history: int | None = None,
    # General
    use_talib: bool = True,
) -> pl.DataFrame:
    """
    Detect order blocks and return a Polars DataFrame with confirmed blocks.
    """
    # Limit history if needed
    if max_history is not None:
        df = df.tail(max_history)

    # Extract numpy arrays from Polars
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    close = df["close"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)
    dates = df["date"].to_numpy()  # keep as original dtype

    # ZigZag peaks/valleys using scipy
    peak_indices, valley_indices = zigzag_peaks_valleys(
        high, low,
        prominence_peak=zigzag_prominence_peak,
        prominence_valley=zigzag_prominence_valley,
        distance=zigzag_distance,
        width=zigzag_width,
        wlen=zigzag_wlen,
        rel_height=zigzag_rel_height,
        plateau_size=zigzag_plateau_size,
    )

    # Pre‑compute all indicators
    indicators = precompute_indicators(
        high, low, close, volume,
        atr_period=atr_period,
        volume_window=volume_window,
        liquidity_window=liquidity_window,
        zone_atr_multiplier=zone_atr_multiplier,
        adx_period=adx_period if use_adx_filter else None,
        rsi_period=rsi_period if use_rsi_confirmation else None,
        macd_fast=macd_fast if use_macd_confirmation else None,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        use_talib=use_talib,
    )

    # Generate candidates
    candidates = generate_block_candidates(
        high, low, close, volume, dates,
        peak_indices, valley_indices, indicators,
        lookback_min=lookback_min,
        lookback_max=lookback_max,
        lookback_atr_multiplier=lookback_atr_multiplier,
        use_dynamic_lookback=use_dynamic_lookback,
        liquidity_tolerance=liquidity_tolerance,
        multiple_breakouts=multiple_breakouts,
        breakout_volume_threshold=breakout_volume_threshold,
        use_adx_filter=use_adx_filter,
        adx_threshold=adx_threshold,
    )

    # Validate
    confirmed = validate_block_candidates(
        high, low, close, volume, dates,
        candidates, indicators,
        confirmation_window=confirmation_window,
        min_reaction_size=min_reaction_size,
        use_rsi_confirmation=use_rsi_confirmation,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        use_macd_confirmation=use_macd_confirmation,
    )

    # Cluster if needed
    if cluster_blocks and confirmed:
        confirmed = cluster_order_blocks(confirmed, cluster_price_tolerance)

    # Convert to Polars DataFrame
    if not confirmed:
        return pl.DataFrame(schema={
            "block_type": pl.Utf8,
            "start": pl.Datetime,
            "break": pl.Datetime,
            "retest": pl.Datetime,
            "zone_low": pl.Float64,
            "zone_high": pl.Float64,
            "strength": pl.Float64,
        })
    data = {
        "block_type": [b.block_type for b in confirmed],
        "start": [b.start for b in confirmed],
        "break": [b.break_ for b in confirmed],
        "retest": [b.retest for b in confirmed],
        "zone_low": [b.zone_low for b in confirmed],
        "zone_high": [b.zone_high for b in confirmed],
        "strength": [b.strength for b in confirmed],
    }
    return pl.DataFrame(data)