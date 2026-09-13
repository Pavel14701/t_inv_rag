"""Smoke test for TZ-12 benchmark runner (marked performance)."""

from __future__ import annotations

import pytest

from ta.benchmarks.run import main, run_benchmarks


@pytest.mark.performance
def test_benchmark_smoke_table() -> None:
    """Runner produces a well-formed markdown table on small input."""
    table = run_benchmarks(3000, repeats=1, only=["sma", "rsi"])
    assert "# ta/ benchmarks (n=3000" in table
    assert "| module | indicator | impl | ms | speedup |" in table
    assert "numba (cold)" in table
    assert "numba (warm)" in table
    assert "pandas" in table


@pytest.mark.performance
def test_benchmark_cli_main() -> None:
    """CLI entry returns 0 without --save."""
    assert main(["--n", "2000", "--repeats", "1"]) == 0