"""Data types used across the trading system.

Defines the core OrderBlock dataclass representing supply/demand zones.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True)
class OrderBlock:
    """An immutable representation of a supply or demand order block.

    An order block is a price zone formed by smart money activity,
    used as a potential entry point.

    Args:
        id: Unique identifier.
        block_type: 'supply' or 'demand'.
        start: Datetime when the block began forming.
        break_: Datetime when the block was broken.
        retest: Datetime of a retest of the block.
        zone_low: Lower boundary of the price zone.
        zone_high: Upper boundary of the price zone.
        strength: A measure of block strength (>= 0).
        structure_label: Optional label, e.g. 'valid', 'broken', 'weak'.
        trend_direction: 'up', 'down', or None.
        start_idx: Index of the first bar (in a dataframe), default -1.
        end_idx: Index of the last bar, default -1.

    """

    id: int
    block_type: str
    start: datetime
    break_: datetime
    retest: datetime
    zone_low: float
    zone_high: float
    strength: float = 0.0
    structure_label: str | None = None
    trend_direction: str | None = None
    start_idx: int = -1
    end_idx: int = -1
