# -*- coding: utf-8 -*-
"""Market structure / order block detection package.

Split out of the former single-module ``market_structure.py``:
- :mod:`config`      - OrderBlockConfig (all tuning parameters)
- :mod:`types`       - OrderBlock dataclass
- :mod:`structure`   - HH/HL/LH/LL market structure classification
- :mod:`indicators`  - indicator pre-computation
- :mod:`candidates`  - breakout candidate generation (Numba)
- :mod:`filters`     - validation sub-filters (FVG, breaker, zones, ...)
- :mod:`validation`  - retest search + look-ahead guards
- :mod:`clustering`  - overlapping-block clustering
- :mod:`online_zigzag` - repaint-free incremental ZigZag
- :mod:`blocks`      - :func:`identify_order_blocks` entry point
- :mod:`configs`     - per-timeframe presets
"""
from .blocks import identify_order_blocks
from .clustering import cluster_order_blocks
from .config import OrderBlockConfig
from .configs import TIMEFRAME_CONFIGS, get_order_block_config
from .candidates import generate_block_candidates
from .indicators import compute_lookback, precompute_indicators
from .online_zigzag import (
    OnlineZigZag,
    Pivot,
    confirmed_pivot_arrays,
    zigzag_reversal_numpy,
)
from .structure import (
    classify_market_structure,
    is_block_aligned_with_trend,
)
from .types import OrderBlock
from .validation import validate_block_candidates

__all__ = [
    'identify_order_blocks',
    'OrderBlock',
    'OrderBlockConfig',
    'OnlineZigZag',
    'Pivot',
    'zigzag_reversal_numpy',
    'confirmed_pivot_arrays',
    'classify_market_structure',
    'is_block_aligned_with_trend',
    'cluster_order_blocks',
    'generate_block_candidates',
    'validate_block_candidates',
    'precompute_indicators',
    'compute_lookback',
    'TIMEFRAME_CONFIGS',
    'get_order_block_config',
]
