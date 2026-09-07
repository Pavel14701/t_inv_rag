# -*- coding: utf-8 -*-
"""Market structure classification (HH/HL/LH/LL and trend direction)."""
from __future__ import annotations

import numpy as np
from numba import float64, int64, int8, njit  # type: ignore[attr-defined]


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
        trend_code = 0   # up
    elif down_streak >= min_consecutive:
        trend_code = 1   # down
    else:
        trend_code = 2   # unknown
    return peak_higher, valley_higher, trend_code


def classify_market_structure(
    peaks: list[int],
    valleys: list[int],
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    lookback: int = 10,
    min_consecutive: int = 3,
) -> tuple[str | None, str | None]:
    """Return (structure_label like 'HH/HL', trend_direction like 'up')."""
    if len(peaks) < 2 or len(valleys) < 2:
        return None, None
    peaks_arr = np.array(peaks, dtype=np.int64)
    valleys_arr = np.array(valleys, dtype=np.int64)
    peak_code, valley_code, trend_code = _classify_market_structure_nb(
        peaks_arr, valleys_arr, high_prices, low_prices, lookback,
        min_consecutive,
    )
    peak_map = {0: 'LH', 1: 'HH', 2: '?'}
    valley_map = {0: 'LL', 1: 'HL', 2: '?'}
    trend_map = {0: 'up', 1: 'down', 2: None}
    peak_label = peak_map.get(peak_code, '?')
    valley_label = valley_map.get(valley_code, '?')
    structure_label = f'{peak_label}/{valley_label}' if '?' not in (
        peak_label, valley_label
    ) else None
    trend_direction = trend_map.get(trend_code)
    return structure_label, trend_direction


def is_block_aligned_with_trend(
    block_type: str, trend_dir: str | None,
) -> bool:
    """True when the block type agrees with the trend direction."""
    if trend_dir is None:
        return True
    if block_type == 'supply' and trend_dir == 'down':
        return True
    if block_type == 'demand' and trend_dir == 'up':
        return True
    return False
