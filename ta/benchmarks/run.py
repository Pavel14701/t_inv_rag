"""Benchmark runner: ``python -m ta.benchmarks.run --n 100000``."""

from __future__ import annotations

import argparse
import time

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ta.src.candle.cdl_engulfing import cdl_engulfing
from ta.src.custom.scrsi import scrsi_ind, scrsi_numpy
from ta.src.momentum.macd import macd_ind, macd_numpy
from ta.src.momentum.rsi import rsi_ind, rsi_numpy
from ta.src.overlap.ema import ema_ind
from ta.src.overlap.sma import sma_ind
from ta.src.volatility.atr import atr_ind


REPEATS = 3


def make_data(
    n: int, seed: int = 42
) -> dict[str, np.ndarray]:
    """Deterministic OHLC data for benchmarks."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    return {
        "open": open_, "high": high, "low": low, "close": close
    }


def _pandas_sma(close: np.ndarray, length: int = 10) -> np.ndarray:
    import pandas as pd

    return pd.Series(close).rolling(length).mean().to_numpy()


def _pandas_ema(close: np.ndarray, length: int = 10) -> np.ndarray:
    import pandas as pd

    return pd.Series(close).ewm(span=length, adjust=False).mean().to_numpy()


def _pandas_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    length: int = 14,
) -> np.ndarray:
    import pandas as pd

    pc = pd.Series(close).shift(1)
    tr = pd.concat([
        pd.Series(high) - pd.Series(low),
        (pd.Series(high) - pc).abs(),
        (pd.Series(low) - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean().to_numpy()


def _talib(name: str) -> Callable | None:
    """Return a TA-Lib callable if installed, else None."""
    try:
        import talib
    except ImportError:
        return None
    fn = getattr(talib, name, None)
    return fn


def _scenarios() -> list[dict[str, Any]]:
    """Scenario definitions: fast core + baselines."""
    talib_sma = _talib("SMA")
    talib_ema = _talib("EMA")
    talib_rsi = _talib("RSI")
    talib_atr = _talib("ATR")
    talib_macd = _talib("MACD")
    talib_cdl = _talib("CDLENGULFING")
    return [
        {
            "module": "overlap", "name": "sma",
            "fast": lambda d: sma_ind(d["close"], length=10),
            "baselines": {
                "pandas": lambda d: _pandas_sma(d["close"], 10),
                "talib": (
                    (lambda d: talib_sma(d["close"], 10))
                    if talib_sma else None
                ),
            },
        },
        {
            "module": "overlap", "name": "ema",
            "fast": lambda d: ema_ind(d["close"], length=10),
            "baselines": {
                "pandas": lambda d: _pandas_ema(d["close"], 10),
                "talib": (
                    (lambda d: talib_ema(d["close"], 10))
                    if talib_ema else None
                ),
            },
        },
        {
            "module": "momentum", "name": "rsi",
            "fast": lambda d: rsi_ind(d["close"], length=14),
            "baselines": {
                "numpy": lambda d: rsi_numpy(d["close"], 14),
                "talib": (
                    (lambda d: talib_rsi(d["close"], 14))
                    if talib_rsi else None
                ),
            },
        },
        {
            "module": "momentum", "name": "macd",
            "fast": lambda d: macd_ind(d["close"]),
            "baselines": {
                "numpy": lambda d: macd_numpy(d["close"]),
                "talib": (
                    (lambda d: talib_macd(d["close"])[0])
                    if talib_macd else None
                ),
            },
        },
        {
            "module": "volatility", "name": "atr",
            "fast": lambda d: atr_ind(
                d["high"], d["low"], d["close"], length=14
            ),
            "baselines": {
                "pandas": lambda d: _pandas_atr(
                    d["high"], d["low"], d["close"], 14
                ),
                "talib": (
                    (lambda d: talib_atr(
                        d["high"], d["low"], d["close"], 14
                    ))
                    if talib_atr else None
                ),
            },
        },
        {
            "module": "candle", "name": "cdl_engulfing",
            "fast": lambda d: cdl_engulfing(
                d["open"], d["high"], d["low"], d["close"]
            ),
            "baselines": {
                "talib": (
                    (lambda d: talib_cdl(
                        d["open"], d["high"], d["low"], d["close"]
                    ))
                    if talib_cdl else None
                ),
            },
        },
        {
            # OTT is skipped (known upstream numba int8+fillna bug);
            # SCRSI is the custom-group representative instead (TZ-12 п.2).
            "module": "custom", "name": "scrsi",
            "fast": lambda d: scrsi_ind(
                d["close"], domcycle=20, vibration=10, leveling=5
            ),
            "baselines": {
                "numpy": lambda d: scrsi_numpy(
                    d["close"], domcycle=20, vibration=10, leveling=5
                )
            },
        },
    ]


SCENARIOS = _scenarios()


def _time_warm(fn: Callable, data: dict, repeats: int) -> float:
    """Best-of-repeats warm timing in ms."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(data)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _time_cold(fn: Callable, data: dict) -> float:
    """First-call timing (includes JIT compilation) in ms."""
    t0 = time.perf_counter()
    fn(data)
    return (time.perf_counter() - t0) * 1000.0


def run_benchmarks(
    n: int, repeats: int = REPEATS, seed: int = 42,
    only: list[str] | None = None,
) -> str:
    """Run all scenarios; return a markdown table."""
    data = make_data(n, seed)
    rows: list[list[str]] = []
    header = (
        "| module | indicator | impl | ms | speedup |"
    )
    sep = "|---|---|---|---:|---:|"
    lines = [f"# ta/ benchmarks (n={n}, repeats={repeats})", "", header, sep]
    for sc in SCENARIOS:
        name = str(sc["name"])
        if only and name not in only:
            continue
        fast: Callable[[dict], object] = sc["fast"]
        baselines: dict[str, Callable | None] = sc["baselines"]
        cold_ms = _time_cold(fast, data)
        fast_ms = _time_warm(fast, data, repeats)
        rows.append([
            str(sc["module"]), name, "numba (cold)", f"{cold_ms:.2f}", "-",
        ])
        rows.append([
            str(sc["module"]), name, "numba (warm)", f"{fast_ms:.2f}", "1.0x",
        ])
        for impl, fn in baselines.items():
            if fn is None:
                rows.append([
                    str(sc["module"]), name, impl, "n/a (not installed)",
                    "-",
                ])
                continue
            base_ms = _time_warm(fn, data, repeats)
            speedup = base_ms / fast_ms if fast_ms > 0 else 0.0
            rows.append([
                str(sc["module"]), name, impl, f"{base_ms:.2f}",
                f"{speedup:.1f}x",
            ])
    for row in rows:
        module, name, impl, ms, speed = row
        lines.append(f"| {module} | {name} | {impl} | {ms} | {speed} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: --n, --repeats, --save."""
    parser = argparse.ArgumentParser(
        prog="ta.benchmarks", description="ta/ numba vs baselines"
    )
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)
    table = run_benchmarks(args.n, repeats=args.repeats)
    print(table)
    if args.save:
        out = Path(__file__).parent / "results"
        out.mkdir(exist_ok=True)
        path = out / f"bench_{args.n}.md"
        path.write_text(table, encoding="utf-8")
        print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
