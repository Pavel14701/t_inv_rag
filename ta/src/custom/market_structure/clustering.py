# -*- coding: utf-8 -*-
"""Order block clustering."""

from __future__ import annotations

from datetime import timedelta

from .types import OrderBlock


def cluster_order_blocks(
    blocks: list[OrderBlock],
    price_tolerance: float,
    max_time_gap: timedelta | None = None,
) -> list[OrderBlock]:
    """Merge overlapping blocks of the same type into one zone."""
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
            time_gap = (
                (b.start - current.start).total_seconds()
                if max_time_gap
                else 0
            )
            if price_dist <= price_tolerance * max(mid_c, 1e-9) and (
                max_time_gap is None
                or time_gap <= max_time_gap.total_seconds()
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
