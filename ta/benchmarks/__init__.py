"""TZ-12 benchmarks: ta/ numba cores vs baselines (pandas naive, TA-Lib).

Honesty rules: cold JIT measured separately from warm timing; identical
inputs; fixed seed; causal windows.
"""

from ta.benchmarks.run import SCENARIOS, main, run_benchmarks


__all__ = ["SCENARIOS", "main", "run_benchmarks"]
