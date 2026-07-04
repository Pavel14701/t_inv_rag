from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class OrderBlock:
    id: int
    block_type: str                 # "supply" или "demand"
    start: datetime
    break_: datetime
    retest: datetime
    zone_low: float
    zone_high: float
    strength: float = 0.0
    structure_label: Optional[str] = None
    trend_direction: Optional[str] = None  # "up", "down", None
    start_idx: int = -1
    end_idx: int = -1
