"""Inference package (TZ-05): стратегия + данные → сигналы (опц. P(win)).

Модули:
- :mod:`infer.provider` — BarSeriesProvider (каузальные ta-индикаторы,
  compute-once + кэш, O(1) offset-индексация);
- :mod:`infer.data` — источники данных (synthetic/parquet/yfinance/tinvest);
- :mod:`infer.engine` — bar-by-bar контур сигналов + сводка;
- :mod:`infer.cli` — CLI.

Read-only по отношению к рынку: никаких брокерских ордеров.
"""

from .engine import InferenceResult, predict_p_win_at, run_inference
from .provider import BarSeriesProvider, WarmupNotReady, build_manifest

__all__ = [
    'InferenceResult',
    'BarSeriesProvider',
    'WarmupNotReady',
    'build_manifest',
    'run_inference',
    'predict_p_win_at',
]