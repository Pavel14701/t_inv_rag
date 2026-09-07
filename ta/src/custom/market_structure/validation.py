# -*- coding: utf-8 -*-
"""Candidate validation: retest search and look-ahead guards."""
from __future__ import annotations

import numpy as np

from .config import OrderBlockConfig
from .filters import (
    check_breaker,
    check_displacement,
    check_fvg,
    check_orderflow_shift,
    check_rsi_macd,
    check_zone_entry,
    compute_reaction,
    compute_strength,
)
from .structure import classify_market_structure, is_block_aligned_with_trend
from .types import OrderBlock


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
    pivot_confirm: dict[int, int] | None = None,
    pivot_next_extreme: dict[int, int] | None = None,
) -> list[OrderBlock]:
    """Validate candidates and search their first valid retest.

    Look-ahead guards (online mode, ``pivot_confirm`` given)
    -------------------------------------------------------
    - the pivot must have been confirmed strictly *before* the breakout
      bar (``confirm_idx < break_idx``): the block was already tradable
      at the breakout, i.e. the underlying extreme could not repaint;
    - the distance from the pivot to the next confirmed extreme must
      exceed ``cfg.min_extreme_gap``;
    - when ``cfg.require_complete_window`` is set, the full retest
      confirmation window must fit inside the history
      (``end_idx < len(close)``).
    """
    confirmed = existing_blocks.copy()
    avg_vol = indicators['avg_volume']
    zone_low_arr = indicators['zone_low']
    zone_high_arr = indicators['zone_high']
    peak_list = sorted(peak_indices.tolist())
    valley_list = sorted(valley_indices.tolist())
    block_id = len(confirmed)
    for cand in candidates:
        idx = cand['idx']
        break_idx = cand['break_idx']
        is_supply = cand['block_type'] == 'supply'
        base_strength = cand.get('strength', 1.0)
        age_candles = break_idx - idx
        if cfg.max_extreme_age > 0 and age_candles > cfg.max_extreme_age:
            continue
        confirm_idx = None
        next_extreme_idx = None
        if pivot_confirm is not None:
            confirm_idx = pivot_confirm.get(idx)
            if confirm_idx is None or confirm_idx >= break_idx:
                # Pivot was not final at the breakout bar -> repainting
                continue
            if pivot_next_extreme is not None:
                next_extreme_idx = pivot_next_extreme.get(idx)
                if next_extreme_idx is not None and next_extreme_idx >= 0:
                    if next_extreme_idx - idx <= cfg.min_extreme_gap:
                        continue
        # Market structure filter
        if cfg.use_market_structure_filter:
            rel_peaks = [p for p in peak_list if p <= idx]
            rel_valleys = [v for v in valley_list if v <= idx]
            struct_label, trend_dir = classify_market_structure(
                rel_peaks, rel_valleys,
                high, low,
                lookback=cfg.structure_lookback,
                min_consecutive=cfg.min_structure_extremes,
            )
            if not is_block_aligned_with_trend(cand['block_type'], trend_dir):
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
            is_supply, close[break_idx], indicators['atr'][break_idx],
            cfg.breaker_require_displacement, cfg.displacement_multiplier,
            cfg.breaker_bonus_multiplier,
        ) if cfg.check_breaker else 1.0
        # Retest loop
        end_idx = min(break_idx + cfg.confirmation_window, len(close))
        if cfg.require_complete_window and end_idx >= len(close):
            continue  # window not fully inside history
        zone_low = zone_low_arr[idx]
        zone_high = zone_high_arr[idx]
        if not (np.isfinite(zone_low) and np.isfinite(zone_high)):
            continue
        if not _find_retest(
            cand, idx, break_idx, end_idx, is_supply, base_strength,
            age_candles, zone_low, zone_high, fvg_present, breaker_bonus,
            struct_label, trend_dir, confirm_idx, next_extreme_idx,
            confirmed, block_id, high, low, close, volume, dates,
            indicators, avg_vol, peak_list, valley_list, cfg,
        ):
            continue
        block_id += 1
    return confirmed


def _find_retest(  # noqa: PLR0913
    cand: dict,
    idx: int,
    break_idx: int,
    end_idx: int,
    is_supply: bool,
    base_strength: float,
    age_candles: int,
    zone_low: float,
    zone_high: float,
    fvg_present: bool,
    breaker_bonus: float,
    struct_label: str | None,
    trend_dir: str | None,
    confirm_idx: int | None,
    next_extreme_idx: int | None,
    confirmed: list[OrderBlock],
    block_id: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    dates: np.ndarray,
    indicators: dict[str, np.ndarray],
    avg_vol: np.ndarray,
    peak_list: list[int],
    valley_list: list[int],
    cfg: OrderBlockConfig,
) -> bool:
    """Search the first valid retest; append the block when found."""
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
            cfg.use_rsi_confirmation, cfg.rsi_overbought,
            cfg.rsi_oversold, cfg.use_macd_confirmation,
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
            close[j], zone_low, zone_high, is_supply,
        )
        # Displacement check
        if not check_displacement(
            reaction_abs, indicators['atr'][j],
            cfg.displacement_multiplier, reaction_pct,
            cfg.min_reaction_size,
        ):
            continue
        # Orderflow shift
        if cfg.check_orderflow_shift:
            if not check_orderflow_shift(
                peak_list, valley_list, high, low, idx, j,
                cfg.shift_lookforward, is_supply,
                cfg.shift_require_extremes,
            ):
                continue
        # Strength
        strength_val = compute_strength(
            base_strength, volume[j], avg_vol[j],
            reaction_abs, indicators['atr'][j], reaction_pct,
            age_candles, fvg_present, breaker_bonus, cfg,
        )
        confirmed.append(OrderBlock(
            id=block_id,
            block_type=cand['block_type'],
            start=cand['start_date'],
            break_=dates[break_idx],
            retest=dates[j],
            zone_low=float(zone_low),
            zone_high=float(zone_high),
            strength=float(strength_val),
            structure_label=struct_label,
            trend_direction=trend_dir,
            extreme_idx=idx,
            confirm_idx=confirm_idx,
            next_extreme_idx=next_extreme_idx,
        ))
        return True  # first retest wins
    return False
