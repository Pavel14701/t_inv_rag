# -*- coding: utf-8 -*-
"""Validation sub-filters: FVG, breaker, zone entry, RSI/MACD,
reaction, displacement, orderflow shift and strength.
"""
from __future__ import annotations

import bisect

import numpy as np

from .config import OrderBlockConfig
from .types import OrderBlock


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
    """Return (fvg_present, continue_validation)."""
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
        candles_vol = [
            volume[break_idx - 1], volume[break_idx], volume[break_idx + 1],
        ]
        if fvg_volume_mode == 'any':
            ok = any(v > fvg_volume_multiplier * avg for v in candles_vol)
        elif fvg_volume_mode == 'center':
            ok = volume[break_idx] > fvg_volume_multiplier * avg
        elif fvg_volume_mode == 'first':
            ok = volume[break_idx - 1] > fvg_volume_multiplier * avg
        elif fvg_volume_mode == 'last':
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
    """Breaker bonus when a prior opposite zone was cleanly broken."""
    if not confirmed_blocks:
        return 1.0
    start = max(0, len(confirmed_blocks) - breaker_lookback)
    for prev in confirmed_blocks[start:]:
        if prev.block_type == ('supply' if is_supply else 'demand'):
            continue
        if not (
            current_zone_high > prev.zone_low
            and current_zone_low < prev.zone_high
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
    """True if bar j enters the zone per the entry mode."""
    zone_span = zone_high - zone_low
    if zone_entry_mode == 'wick':
        if is_supply:
            if not (zone_low <= low[j] <= zone_high):
                return False
            if low[j] < zone_low - max_zone_penetration * zone_span:
                return False
        else:
            if not (zone_low <= high[j] <= zone_high):
                return False
            if high[j] > zone_high + max_zone_penetration * zone_span:
                return False
        return True
    elif zone_entry_mode == 'close':
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
    """RSI/MACD confirmation gate for the retest bar."""
    if use_rsi:
        rsi = indicators['rsi'][j]
        if not np.isfinite(rsi):
            return False
        if is_supply and rsi < rsi_overbought:
            return False
        if not is_supply and rsi > rsi_oversold:
            return False
    if use_macd:
        hist = indicators['macd_hist'][j]
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
    """Reaction (absolute, pct) of the retest close beyond the zone."""
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
    """True when the retest reaction is strong enough."""
    if displacement_multiplier > 0:
        if not np.isfinite(atr_val) or (
            reaction_abs < displacement_multiplier * atr_val
        ):
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
    """True when structure shifted in the block direction."""
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
            return (
                last_valley is not None
                and low[first_idx] < low[last_valley]
            )
        else:
            return (
                last_valley is not None
                and low[first_idx] > low[last_valley]
            )


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
    """Composite strength score of a confirmed block."""
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
