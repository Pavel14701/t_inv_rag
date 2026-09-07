# -*- coding: utf-8 -*-
"""Data types of the order block pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class OrderBlock:
    """Represents a confirmed order block.

    The ``extreme_idx`` / ``confirm_idx`` / ``next_extreme_idx`` fields
    are only populated in the online (repaint-free) mode: they expose
    the bar of the underlying pivot, the bar at which that pivot became
    final and the bar of the following extreme - the values needed for
    look-ahead audits.
    """

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
    extreme_idx: int | None = None     # bar of the pivot (online mode)
    confirm_idx: int | None = None     # bar the pivot became final
    next_extreme_idx: int | None = None  # bar of the following extreme
