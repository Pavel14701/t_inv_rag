# -*- coding: utf-8 -*-
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import polars as pl
from numba import boolean, float64, int8, int64, njit  # type: ignore[attr-defined]
from numba.typed import List

from ..momentum import macd_ind, rsi_ind
from ..overlap import sma_ind
from ..trend import adx_ind, zigzag_peaks_valleys
from ..utils import _rolling_max_numba, _rolling_min_numba
from ..volatility import atr_ind


# ----------------------------------------------------------------------
# Configuration (final version with all tuning parameters)
# ----------------------------------------------------------------------
@dataclass
class OrderBlockConfig:
    # ZigZag
    zigzag_prominence_peak: float = 0.01
    zigzag_prominence_valley: float = 0.01
    zigzag_distance: int = 5
    zigzag_width: float | None = None
    zigzag_wlen: int | None = None
    zigzag_rel_height: float = 0.5
    zigzag_plateau_size: int | None = None

    # Breakout & lookback
    lookback_min: int = 5
    lookback_max: int = 50
    lookback_atr_multiplier: float = 2.0
    use_dynamic_lookback: bool = True
    multiple_breakouts: bool = False
    breakout_volume_threshold: float = 1.0
    breakout_impulse_multiplier: float = 0.0

    # Trend filter (ADX)
    use_adx_filter: bool = False
    adx_period: int = 14
    adx_threshold: float = 25.0

    # Market structure filter (HH/HL/LH/LL)
    use_market_structure_filter: bool = False
    structure_lookback: int = 10
    min_structure_extremes: int = 3

    # Freshness filter
    max_extreme_age: int = 0

    # Fair Value Gap (FVG) filters
    require_fvg: bool = False
    fvg_tolerance: float = 0.0
    fvg_volume_multiplier: float = 0.0
    fvg_volume_mode: Literal["any", "center", "first", "last"] = "any"
    fvg_bonus_multiplier: float = 1.0

    # Breaker block filter
    check_breaker: bool = False
    breaker_lookback: int = 50
    breaker_bonus_multiplier: float = 1.5
    breaker_require_displacement: bool = False
    breaker_check_mitigated: bool = False

    # Orderflow shift filter
    check_orderflow_shift: bool = False
    shift_lookforward: int = 10
    shift_require_extremes: bool = True

    # Zone entry mode
    zone_entry_mode: Literal["wick", "close", "any"] = "wick"

    # Mitigation / closure filter
    require_closure_outside: bool = False

    # Displacement (strong reaction) filter
    displacement_multiplier: float = 0.0

    # Additional confirmation (RSI, MACD)
    use_rsi_confirmation: bool = False
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    use_macd_confirmation: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Retest & reaction
    confirmation_window: int = 10
    min_reaction_size: float = 0.002
    max_zone_penetration: float = 0.5

    # Zone sizing
    atr_period: int = 14
    zone_atr_multiplier: float = 1.0

    # Volume & liquidity
    volume_window: int = 20
    liquidity_window: int = 10
    liquidity_tolerance: float = 0.001

    # Strength calculation options
    strength_log_volume: bool = False
    strength_atr_normalize: bool = False
    strength_age_penalty: bool = False
    strength_age_halflife: int = 20
    strength_max_multiplier: float = 10.0
    strength_reaction_cap: float = 3.0   # maximum allowed reaction factor

    # Clustering
    cluster_blocks: bool = False
    cluster_price_tolerance: float = 0.001
    max_cluster_time_gap: timedelta | None = timedelta(days=30)

    # Historical depth
    max_history: int | None = None

    # General
    use_talib: bool = True

    def __post_init__(self):
        if self.strength_reaction_cap <= 0:
            raise ValueError(
                f"strength_reaction_cap must be positive; \
                    got {self.strength_reaction_cap}"
            )
        if self.strength_reaction_cap > self.strength_max_multiplier:
            raise ValueError(
                "strength_reaction_cap must not exceed strength_max_multiplier; "
                f"got strength_reaction_cap={self.strength_reaction_cap}, "
                f"strength_max_multiplier={self.strength_max_multiplier}"
            )


@dataclass
class OrderBlock:
    """Represents a confirmed order block."""
    id: int
    block_type: str          # "supply" or "demand"
    start: datetime          # time of the original peak/valley
    break_: datetime         # time of the breakout candle
    retest: datetime         # time of the retest candle
    zone_low: float          # lower bound of the zone
    zone_high: float         # upper bound of the zone
    strength: float = 0.0    # optional strength score
    structure_label: str | None = None
    trend_direction: str | None = None


# ----------------------------------------------------------------------
# Market structure helpers (Numba-optimized)
# ----------------------------------------------------------------------
@njit((int64[:], int64[:], float64[:], float64[:], int64, int64),
      cache=True, fastmath=True)
def _classify_market_structure_nb(
    peaks, valleys, high, low, lookback, min_consecutive
) -> tuple[int, int, int]:
    n_peaks = len(peaks)
    n_valleys = len(valleys)
    start_peaks = max(0, n_peaks - lookback)
    start_valleys = max(0, n_valleys - lookback)
    rec_peaks = peaks[start_peaks:]
    rec_valleys = valleys[start_valleys:]
    # Peak direction: 1 = HH, 0 = LH, 2 = unknown
    if len(rec_peaks) >= 2:
        last_peak = high[rec_peaks[-1]]
        prev_peak = high[rec_peaks[-2]]
        peak_higher = 1 if last_peak > prev_peak else 0
    else:
        peak_higher = 2
    # Valley direction: 1 = HL, 0 = LL, 2 = unknown
    if len(rec_valleys) >= 2:
        last_valley = low[rec_valleys[-1]]
        prev_valley = low[rec_valleys[-2]]
        valley_higher = 1 if last_valley > prev_valley else 0
    else:
        valley_higher = 2
    total_len = len(rec_peaks) + len(rec_valleys)
    if total_len == 0:
        return 2, 2, 2
    # Merge two sorted lists of types
    combined_types = np.empty(total_len, dtype=int8)
    i = j = k = 0
    while i < len(rec_peaks) and j < len(rec_valleys):
        if rec_peaks[i] < rec_valleys[j]:
            combined_types[k] = 1   # peak
            i += 1
        else:
            combined_types[k] = 0   # valley
            j += 1
        k += 1
    while i < len(rec_peaks):
        combined_types[k] = 1
        i += 1
        k += 1
    while j < len(rec_valleys):
        combined_types[k] = 0
        j += 1
        k += 1
    # Count streak of last min_consecutive types
    start = max(0, total_len - min_consecutive)
    up_streak = 0
    down_streak = 0
    for idx in range(start, total_len):
        t = combined_types[idx]
        if t == 1:  # peak
            if peak_higher == 1:
                up_streak += 1
            elif peak_higher == 0:
                down_streak += 1
        else:       # valley
            if valley_higher == 1:
                up_streak += 1
            elif valley_higher == 0:
                down_streak += 1
    if up_streak >= min_consecutive:
        trend_dir = 0   # up
    elif down_streak >= min_consecutive:
        trend_dir = 1   # down
    else:
        trend_dir = 2   # none
    return peak_higher, valley_higher, trend_dir


def classify_market_structure(
    peaks: list[int],
    valleys: list[int],
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    lookback: int = 10,
    min_consecutive: int = 3
) -> tuple[str | None, str | None]:
    if len(peaks) < 2 or len(valleys) < 2:
        return None, None
    peaks_arr = np.array(peaks, dtype=np.int64)
    valleys_arr = np.array(valleys, dtype=np.int64)
    peak_code, valley_code, trend_code = _classify_market_structure_nb(
        peaks_arr, valleys_arr, high_prices, low_prices, lookback, min_consecutive
    )
    peak_map = {0: "LH", 1: "HH", 2: "?"}
    valley_map = {0: "LL", 1: "HL", 2: "?"}
    trend_map = {0: "up", 1: "down", 2: None}
    peak_label = peak_map.get(peak_code, "?")
    valley_label = valley_map.get(valley_code, "?")
    structure_label = f"{peak_label}/{valley_label}" if "?" not in (
        peak_label, valley_label
    ) else None
    trend_direction = trend_map.get(trend_code)
    return structure_label, trend_direction


def is_block_aligned_with_trend(block_type: str, trend_dir: str | None) -> bool:
    if trend_dir is None:
        return True
    if block_type == "supply" and trend_dir == "down":
        return True
    if block_type == "demand" and trend_dir == "up":
        return True
    return False


# ----------------------------------------------------------------------
# Indicator pre‑computation
# ----------------------------------------------------------------------
def precompute_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    cfg: OrderBlockConfig,
) -> dict[str, np.ndarray]:
    atr = atr_ind(high, low, close, length=cfg.atr_period, use_talib=cfg.use_talib)
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
            high, low, close,
            length=cfg.adx_period,
            use_talib=cfg.use_talib,
        )
        result["adx"] = adx
        result["di_plus"] = di_plus
        result["di_minus"] = di_minus
    if cfg.use_rsi_confirmation:
        result["rsi"] = rsi_ind(close, length=cfg.rsi_period, use_talib=cfg.use_talib)
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


# ----------------------------------------------------------------------
# Candidate generation (Numba-optimized)
# ----------------------------------------------------------------------
@njit((float64[:], float64[:], float64[:], float64[:],
       float64[:], float64[:], float64[:], float64[:],
       boolean[:], boolean[:],
       float64[:], float64[:], float64[:],
       int64, int64, float64, float64, float64, boolean, float64, boolean),
      cache=True, fastmath=True)
def _generate_block_candidates_nb(
    high, low, close, volume,
    avg_volume, atr, local_highs, local_lows,
    peak_mask, valley_mask,
    adx, di_plus, di_minus,
    lookback, lookback_max,
    breakout_volume_threshold,
    breakout_impulse_multiplier,
    liquidity_tolerance,
    use_adx_filter,
    adx_threshold,
    multiple_breakouts,
):
    n = len(high)
    block_types = List.empty_list(int64)
    idx_list = List.empty_list(int64)
    break_idx_list = List.empty_list(int64)
    strength_list = List.empty_list(float64)
    for i in range(lookback, n):
        idx = i - lookback
        if idx < 0:
            continue
        if peak_mask[idx]:
            if abs(local_highs[idx] - high[idx]) >= liquidity_tolerance:
                continue
            if multiple_breakouts:
                max_future = min(i + lookback_max, n)
                for brk in range(i, max_future):
                    if low[brk] < low[idx]:
                        ok = True
                        if volume[brk] <= breakout_volume_threshold * avg_volume[brk]:
                            ok = False
                        if ok and breakout_impulse_multiplier > 0:
                            candle_range = high[brk] - low[brk]
                            if not np.isfinite(
                                atr[brk]
                            ) or candle_range < breakout_impulse_multiplier * atr[brk]:
                                ok = False
                        if ok and use_adx_filter:
                            if not np.isfinite(adx[brk]) or adx[brk] < adx_threshold:
                                ok = False
                            elif di_minus[brk] <= di_plus[brk]:
                                ok = False
                        if ok:
                            move = (close[idx] - close[brk]) / max(close[idx], 1e-9)
                            strength = max(0.0, move)
                            block_types.append(0)  # supply
                            idx_list.append(idx)
                            break_idx_list.append(brk)
                            strength_list.append(strength)
                            break
            else:
                if low[i] < low[idx]:
                    ok = True
                    if volume[i] <= breakout_volume_threshold * avg_volume[i]:
                        ok = False
                    if ok and breakout_impulse_multiplier > 0:
                        candle_range = high[i] - low[i]
                        if not np.isfinite(
                            atr[i]
                        ) or candle_range < breakout_impulse_multiplier * atr[i]:
                            ok = False
                    if ok and use_adx_filter:
                        if not np.isfinite(adx[i]) or adx[i] < adx_threshold:
                            ok = False
                        elif di_minus[i] <= di_plus[i]:
                            ok = False
                    if ok:
                        block_types.append(0)
                        idx_list.append(idx)
                        break_idx_list.append(i)
                        strength_list.append(1.0)
        elif valley_mask[idx]:
            if abs(local_lows[idx] - low[idx]) >= liquidity_tolerance:
                continue
            if multiple_breakouts:
                max_future = min(i + lookback_max, n)
                for brk in range(i, max_future):
                    if high[brk] > high[idx]:
                        ok = True
                        if volume[brk] <= breakout_volume_threshold * avg_volume[brk]:
                            ok = False
                        if ok and breakout_impulse_multiplier > 0:
                            candle_range = high[brk] - low[brk]
                            if not np.isfinite(
                                atr[brk]
                            ) or candle_range < breakout_impulse_multiplier * atr[brk]:
                                ok = False
                        if ok and use_adx_filter:
                            if not np.isfinite(adx[brk]) or adx[brk] < adx_threshold:
                                ok = False
                            elif di_plus[brk] <= di_minus[brk]:
                                ok = False
                        if ok:
                            move = (close[brk] - close[idx]) / max(close[idx], 1e-9)
                            strength = max(0.0, move)
                            block_types.append(1)  # demand
                            idx_list.append(idx)
                            break_idx_list.append(brk)
                            strength_list.append(strength)
                            break
            else:
                if high[i] > high[idx]:
                    ok = True
                    if volume[i] <= breakout_volume_threshold * avg_volume[i]:
                        ok = False
                    if ok and breakout_impulse_multiplier > 0:
                        candle_range = high[i] - low[i]
                        if not np.isfinite(atr[i]) or candle_range < breakout_impulse_multiplier * atr[i]:  # noqa: E501
                            ok = False
                    if ok and use_adx_filter:
                        if not np.isfinite(adx[i]) or adx[i] < adx_threshold:
                            ok = False
                        elif di_plus[i] <= di_minus[i]:
                            ok = False
                    if ok:
                        block_types.append(1)
                        idx_list.append(idx)
                        break_idx_list.append(i)
                        strength_list.append(1.0)
    return block_types, idx_list, break_idx_list, strength_list


def generate_block_candidates(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    dates: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    indicators: dict[str, np.ndarray],
    cfg: OrderBlockConfig,
) -> list[dict]:
    n = len(high)
    lookback = _compute_lookback(indicators, cfg)
    # Masks
    peak_mask = np.zeros(n, dtype=bool)
    valley_mask = np.zeros(n, dtype=bool)
    if len(peak_indices) > 0:
        peak_mask[peak_indices] = True
    if len(valley_indices) > 0:
        valley_mask[valley_indices] = True
    avg_volume = indicators["avg_volume"]
    atr = indicators["atr"]
    local_highs = indicators["local_highs"]
    local_lows = indicators["local_lows"]
    # ADX arrays (or placeholders)
    if cfg.use_adx_filter and "adx" in indicators:
        adx = indicators["adx"]
        di_plus = indicators["di_plus"]
        di_minus = indicators["di_minus"]
    else:
        adx = np.full(n, np.nan, dtype=np.float64)
        di_plus = np.full(n, np.nan, dtype=np.float64)
        di_minus = np.full(n, np.nan, dtype=np.float64)
    block_types, idx_list, break_idx_list, strength_list = _generate_block_candidates_nb(  # noqa: E501
        high, low, close, volume,
        avg_volume, atr, local_highs, local_lows,
        peak_mask, valley_mask,
        adx, di_plus, di_minus,
        lookback, cfg.lookback_max,
        cfg.breakout_volume_threshold,
        cfg.breakout_impulse_multiplier,
        cfg.liquidity_tolerance,
        cfg.use_adx_filter,
        cfg.adx_threshold,
        cfg.multiple_breakouts,
    )
    candidates = []
    for bt, idx, brk, st in zip(block_types, idx_list, break_idx_list, strength_list):
        block_type = "supply" if bt == 0 else "demand"
        candidates.append({
            "block_type": block_type,
            "idx": idx,
            "break_idx": brk,
            "strength": st,
            "start_date": dates[idx],
        })
    return candidates


def _compute_lookback(indicators: dict[str, np.ndarray], cfg: OrderBlockConfig) -> int:
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


# ----------------------------------------------------------------------
# Validation subfunctions (optimized with bisect)
# ----------------------------------------------------------------------
def check_fvg(
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    avg_vol: np.ndarray,
    break_idx: int,
    is_supply: bool,
    require_fvg: bool,
    fvg_tolerance: float,
    fvg_volume_multiplier: float,
    fvg_volume_mode: str,
) -> tuple[bool, bool]:
    if not require_fvg:
        return False, True
    n = len(high)
    if break_idx < 1 or break_idx >= n - 1:
        return False, False
    gap_range = abs(high[break_idx - 1] - low[break_idx + 1])
    allowed = fvg_tolerance * gap_range
    if is_supply:
        if high[break_idx + 1] < low[break_idx - 1] + allowed:
            fvg_present = True
        else:
            return False, False
    else:
        if low[break_idx + 1] > high[break_idx - 1] - allowed:
            fvg_present = True
        else:
            return False, False
    if fvg_volume_multiplier > 0:
        avg = avg_vol[break_idx]
        candles_vol = [volume[break_idx - 1], volume[break_idx], volume[break_idx + 1]]
        if fvg_volume_mode == "any":
            ok = any(v > fvg_volume_multiplier * avg for v in candles_vol)
        elif fvg_volume_mode == "center":
            ok = volume[break_idx] > fvg_volume_multiplier * avg
        elif fvg_volume_mode == "first":
            ok = volume[break_idx - 1] > fvg_volume_multiplier * avg
        elif fvg_volume_mode == "last":
            ok = volume[break_idx + 1] > fvg_volume_multiplier * avg
        else:
            ok = any(v > fvg_volume_multiplier * avg for v in candles_vol)
        if not ok:
            return False, False
    return fvg_present, True


def check_breaker(
    confirmed_blocks: list[OrderBlock],
    breaker_lookback: int,
    current_zone_low: float,
    current_zone_high: float,
    is_supply: bool,
    close_break: float,
    atr_break: float,
    breaker_require_displacement: bool,
    displacement_multiplier: float,
    breaker_bonus_multiplier: float,
) -> float:
    if not confirmed_blocks:
        return 1.0
    start = max(0, len(confirmed_blocks) - breaker_lookback)
    for prev in confirmed_blocks[start:]:
        if prev.block_type == ("supply" if is_supply else "demand"):
            continue
        if not (
            current_zone_high > prev.zone_low and current_zone_low < prev.zone_high
        ):
            continue
        if is_supply and close_break < prev.zone_low:
            if breaker_require_displacement:
                move = prev.zone_low - close_break
                if move < displacement_multiplier * atr_break:
                    continue
            return breaker_bonus_multiplier * (1 + prev.strength)
        elif not is_supply and close_break > prev.zone_high:
            if breaker_require_displacement:
                move = close_break - prev.zone_high
                if move < displacement_multiplier * atr_break:
                    continue
            return breaker_bonus_multiplier * (1 + prev.strength)
    return 1.0


def check_zone_entry(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    j: int,
    zone_low: float,
    zone_high: float,
    is_supply: bool,
    max_zone_penetration: float,
    zone_entry_mode: str,
) -> bool:
    if zone_entry_mode == "wick":
        if is_supply:
            if not (zone_low <= low[j] <= zone_high):
                return False
            if low[j] < zone_low - max_zone_penetration * (zone_high - zone_low):
                return False
        else:
            if not (zone_low <= high[j] <= zone_high):
                return False
            if high[j] > zone_high + max_zone_penetration * (zone_high - zone_low):
                return False
        return True
    elif zone_entry_mode == "close":
        if is_supply:
            if not (zone_low <= close[j] <= zone_high):
                return False
        else:
            if not (zone_low <= close[j] <= zone_high):
                return False
        return True
    else:  # "any"
        if is_supply:
            return (
                zone_low <= low[j] <= zone_high
            ) or (
                zone_low <= close[j] <= zone_high
            )
        else:
            return (
                zone_low <= high[j] <= zone_high
            ) or (
                zone_low <= close[j] <= zone_high
            )


def check_rsi_macd(
    indicators: dict[str, np.ndarray],
    j: int,
    is_supply: bool,
    use_rsi: bool,
    rsi_overbought: float,
    rsi_oversold: float,
    use_macd: bool,
) -> bool:
    if use_rsi:
        rsi = indicators["rsi"][j]
        if not np.isfinite(rsi):
            return False
        if is_supply and rsi < rsi_overbought:
            return False
        if not is_supply and rsi > rsi_oversold:
            return False
    if use_macd:
        hist = indicators["macd_hist"][j]
        if not np.isfinite(hist):
            return False
        if is_supply and hist > 0:
            return False
        if not is_supply and hist < 0:
            return False
    return True


def compute_reaction(
    close_j: float,
    zone_low: float,
    zone_high: float,
    is_supply: bool,
) -> tuple[float, float]:
    if is_supply:
        ref_price = zone_high
        reaction_abs = ref_price - close_j
        reaction_pct = reaction_abs / max(ref_price, 1e-9)
    else:
        ref_price = zone_low
        reaction_abs = close_j - ref_price
        reaction_pct = reaction_abs / max(ref_price, 1e-9)
    return reaction_abs, reaction_pct


def check_displacement(
    reaction_abs: float,
    atr_val: float,
    displacement_multiplier: float,
    reaction_pct: float,
    min_reaction_size: float,
) -> bool:
    if displacement_multiplier > 0:
        if not np.isfinite(atr_val) or reaction_abs < displacement_multiplier * atr_val:
            return False
    else:
        if reaction_pct < min_reaction_size:
            return False
    return True


def check_orderflow_shift(
    peak_list: list[int],
    valley_list: list[int],
    high: np.ndarray,
    low: np.ndarray,
    idx: int,
    j: int,
    shift_lookforward: int,
    is_supply: bool,
    shift_require_extremes: bool,
) -> bool:
    future_end = j + shift_lookforward
    start_peaks = bisect.bisect_right(peak_list, j)
    end_peaks = bisect.bisect_left(peak_list, future_end)
    future_peaks = peak_list[start_peaks:end_peaks]
    start_valleys = bisect.bisect_right(valley_list, j)
    end_valleys = bisect.bisect_left(valley_list, future_end)
    future_valleys = valley_list[start_valleys:end_valleys]
    if not future_peaks and not future_valleys:
        return not shift_require_extremes
    all_future = future_peaks + future_valleys
    first_idx = min(all_future)
    is_peak = first_idx in future_peaks
    pos_peak = bisect.bisect_left(peak_list, idx) - 1
    last_peak = peak_list[pos_peak] if pos_peak >= 0 else None
    pos_valley = bisect.bisect_left(valley_list, idx) - 1
    last_valley = valley_list[pos_valley] if pos_valley >= 0 else None
    if is_peak:
        if is_supply:
            return last_peak is not None and high[first_idx] < high[last_peak]
        else:
            return last_peak is not None and high[first_idx] > high[last_peak]
    else:
        if is_supply:
            return last_valley is not None and low[first_idx] < low[last_valley]
        else:
            return last_valley is not None and low[first_idx] > low[last_valley]


def compute_strength(
    base_strength: float,
    volume_j: float,
    avg_vol_j: float,
    reaction_abs: float,
    atr_j: float,
    reaction_pct: float,
    age_candles: int,
    fvg_present: bool,
    breaker_bonus: float,
    cfg: OrderBlockConfig,
) -> float:
    if cfg.strength_log_volume:
        vol_factor = np.log1p(volume_j / max(avg_vol_j, 1e-9))
    else:
        vol_factor = volume_j / max(avg_vol_j, 1e-9)
    if cfg.strength_atr_normalize:
        reaction_factor = reaction_abs / max(atr_j, 1e-9)
    else:
        reaction_factor = reaction_pct
    # Apply configurable cap
    reaction_factor = min(reaction_factor, cfg.strength_reaction_cap)
    age_factor = 1.0
    if cfg.strength_age_penalty and cfg.strength_age_halflife > 0:
        age_factor = 0.5 ** (age_candles / cfg.strength_age_halflife)
    fvg_bonus = cfg.fvg_bonus_multiplier if fvg_present else 1.0
    total_multiplier = vol_factor * reaction_factor * age_factor * fvg_bonus * breaker_bonus  # noqa: E501
    total_multiplier = min(total_multiplier, cfg.strength_max_multiplier)
    strength = base_strength * total_multiplier
    return max(0.0, strength)


# ----------------------------------------------------------------------
# Validation and retest (final version)
# ----------------------------------------------------------------------
def validate_block_candidates(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    dates: np.ndarray,
    candidates: list[dict],
    indicators: dict[str, np.ndarray],
    cfg: OrderBlockConfig,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
    existing_blocks: list[OrderBlock],
) -> list[OrderBlock]:
    confirmed = existing_blocks.copy()
    avg_vol = indicators["avg_volume"]
    zone_low_arr = indicators["zone_low"]
    zone_high_arr = indicators["zone_high"]
    peak_list = sorted(peak_indices.tolist())
    valley_list = sorted(valley_indices.tolist())
    block_id = len(confirmed)
    for cand in candidates:
        idx = cand["idx"]
        break_idx = cand["break_idx"]
        is_supply = cand["block_type"] == "supply"
        base_strength = cand.get("strength", 1.0)
        age_candles = break_idx - idx
        if cfg.max_extreme_age > 0 and age_candles > cfg.max_extreme_age:
            continue
        # Market structure filter
        if cfg.use_market_structure_filter:
            rel_peaks = [p for p in peak_list if p <= idx]
            rel_valleys = [v for v in valley_list if v <= idx]
            struct_label, trend_dir = classify_market_structure(
                rel_peaks, rel_valleys,
                high, low,
                lookback=cfg.structure_lookback,
                min_consecutive=cfg.min_structure_extremes
            )
            if not is_block_aligned_with_trend(cand["block_type"], trend_dir):
                continue
        else:
            struct_label, trend_dir = None, None
        # FVG
        fvg_present, cont = check_fvg(
            high, low, volume, avg_vol, break_idx, is_supply,
            cfg.require_fvg, cfg.fvg_tolerance,
            cfg.fvg_volume_multiplier, cfg.fvg_volume_mode,
        )
        if not cont:
            continue
        # Breaker bonus
        breaker_bonus = check_breaker(
            confirmed, cfg.breaker_lookback,
            zone_low_arr[idx], zone_high_arr[idx],
            is_supply, close[break_idx], indicators["atr"][break_idx],
            cfg.breaker_require_displacement, cfg.displacement_multiplier,
            cfg.breaker_bonus_multiplier,
        ) if cfg.check_breaker else 1.0
        # Retest loop
        end_idx = min(break_idx + cfg.confirmation_window, len(close))
        zone_low = zone_low_arr[idx]
        zone_high = zone_high_arr[idx]
        if not (np.isfinite(zone_low) and np.isfinite(zone_high)):
            continue
        for j in range(break_idx + 1, end_idx):
            # Zone entry
            if not check_zone_entry(
                high, low, close, j, zone_low, zone_high, is_supply,
                cfg.max_zone_penetration, cfg.zone_entry_mode,
            ):
                continue
            # Volume at retest
            if volume[j] <= avg_vol[j]:
                continue
            # RSI/MACD
            if not check_rsi_macd(
                indicators, j, is_supply,
                cfg.use_rsi_confirmation, cfg.rsi_overbought, cfg.rsi_oversold,
                cfg.use_macd_confirmation,
            ):
                continue
            # Closure outside
            if cfg.require_closure_outside:
                if is_supply and close[j] >= zone_low:
                    continue
                if not is_supply and close[j] <= zone_high:
                    continue
            # Reaction
            reaction_abs, reaction_pct = compute_reaction(
                close[j], 
                zone_low, 
                zone_high, 
                is_supply
            )
            # Displacement check
            if not check_displacement(
                reaction_abs, indicators["atr"][j],
                cfg.displacement_multiplier, reaction_pct, cfg.min_reaction_size,
            ):
                continue
            # Orderflow shift
            if cfg.check_orderflow_shift:
                if not check_orderflow_shift(
                    peak_list, valley_list, high, low, idx, j,
                    cfg.shift_lookforward, is_supply, cfg.shift_require_extremes,
                ):
                    continue
            # Strength
            strength_val = compute_strength(
                base_strength, volume[j], avg_vol[j],
                reaction_abs, indicators["atr"][j], reaction_pct,
                age_candles, fvg_present, breaker_bonus, cfg,
            )
            block = OrderBlock(
                id=block_id,
                block_type=cand["block_type"],
                start=cand["start_date"],
                break_=dates[break_idx],
                retest=dates[j],
                zone_low=float(zone_low),
                zone_high=float(zone_high),
                strength=float(strength_val),
                structure_label=struct_label,
                trend_direction=trend_dir,
            )
            confirmed.append(block)
            block_id += 1
            break  # first retest wins
    return confirmed


# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------
def cluster_order_blocks(
    blocks: list[OrderBlock],
    price_tolerance: float,
    max_time_gap: timedelta | None = None,
) -> list[OrderBlock]:
    if not blocks:
        return blocks
    blocks_sorted = sorted(
        blocks,
        key=lambda b: (b.block_type, (b.zone_low + b.zone_high) / 2, b.start),
    )
    merged = []
    current = blocks_sorted[0]
    for b in blocks_sorted[1:]:
        if b.block_type == current.block_type:
            mid_b = (b.zone_low + b.zone_high) / 2
            mid_c = (current.zone_low + current.zone_high) / 2
            price_dist = abs(mid_b - mid_c)
            time_gap = (b.start - current.start).total_seconds() if max_time_gap else 0
            if price_dist <= price_tolerance * max(mid_c, 1e-9) and (
                max_time_gap is None or time_gap <= max_time_gap.total_seconds()
            ):
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
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    date_col: str = "date",
    cfg: OrderBlockConfig | None = None,
) -> pl.DataFrame:
    if cfg is None:
        cfg = OrderBlockConfig()
    if cfg.max_history is not None:
        df = df.tail(cfg.max_history)
    high = df[high_col].to_numpy().astype(np.float64)
    low = df[low_col].to_numpy().astype(np.float64)
    close = df[close_col].to_numpy().astype(np.float64)
    volume = df[volume_col].to_numpy().astype(np.float64)
    dates = df[date_col].to_numpy()
    peak_indices, valley_indices = zigzag_peaks_valleys(
        high, low,
        prominence_peak=cfg.zigzag_prominence_peak,
        prominence_valley=cfg.zigzag_prominence_valley,
        distance=cfg.zigzag_distance,
        width=cfg.zigzag_width,
        wlen=cfg.zigzag_wlen,
        rel_height=cfg.zigzag_rel_height,
        plateau_size=cfg.zigzag_plateau_size,
    )
    indicators = precompute_indicators(high, low, close, volume, cfg)
    candidates = generate_block_candidates(
        high, low, close, volume, dates,
        peak_indices, valley_indices, indicators, cfg,
    )
    confirmed = validate_block_candidates(
        high, low, close, volume, dates,
        candidates, indicators, cfg,
        peak_indices, valley_indices, [],
    )
    if cfg.cluster_blocks and confirmed:
        confirmed = cluster_order_blocks(
            confirmed,
            cfg.cluster_price_tolerance,
            cfg.max_cluster_time_gap,
        )
    confirmed.sort(key=lambda b: b.start)
    if not confirmed:
        return pl.DataFrame(
            schema={
                "id": pl.Int64,
                "block_type": pl.Utf8,
                "start": pl.Datetime,
                "break": pl.Datetime,
                "retest": pl.Datetime,
                "zone_low": pl.Float64,
                "zone_high": pl.Float64,
                "strength": pl.Float64,
                "structure_label": pl.Utf8,
                "trend_direction": pl.Utf8,
            }
        )
    data = {
        "id": [b.id for b in confirmed],
        "block_type": [b.block_type for b in confirmed],
        "start": [b.start for b in confirmed],
        "break": [b.break_ for b in confirmed],
        "retest": [b.retest for b in confirmed],
        "zone_low": [b.zone_low for b in confirmed],
        "zone_high": [b.zone_high for b in confirmed],
        "strength": [b.strength for b in confirmed],
        "structure_label": [b.structure_label for b in confirmed],
        "trend_direction": [b.trend_direction for b in confirmed],
    }
    out = pl.DataFrame(data).sort("start")
    return out