"""Inference package (TZ-05): strategy + data -> signals (optional P(win)).

Modules:
- :mod:`infer.provider` -- BarSeriesProvider (causal ta indicators,
  compute-once + cache, O(1) offset indexing);
- :mod:`infer.data` -- data sources (synthetic/parquet/yfinance/tinvest);
- :mod:`infer.engine` -- bar-by-bar signal contour + summary;
- :mod:`infer.cli` -- CLI.

Read-only with respect to the market: no broker orders.
"""

from .engine import InferenceResult, predict_p_win_at, run_inference
from .provider import BarSeriesProvider, WarmupNotReady, build_manifest


__all__ = [
    "BarSeriesProvider",
    "InferenceResult",
    "WarmupNotReady",
    "build_manifest",
    "predict_p_win_at",
    "run_inference",
]
