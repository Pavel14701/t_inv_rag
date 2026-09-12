"""Bar-by-bar inference engine (TZ-05 item 3.2).

Signal contour: DSL entry/exit per bar via ``dsl.evaluate_dsl``
(parse once, Interpreter per bar). P(win) filter goes through the
TZ-06 contract (see the ml part at the bottom of this file).
Read-only with respect to the market: no orders.
"""

from __future__ import annotations

import time as _time

from dataclasses import dataclass, field

import polars as pl

from dsl.context import Context
from dsl.exceptions import DSLError, ProviderError
from dsl.interpreter import Interpreter
from dsl.parser import parse

from .provider import BarSeriesProvider, WarmupNotReady


@dataclass
class InferenceResult:
    """Run result: signal series + summary (TZ-05 item 3.3).

    Attributes:
        signals: DataFrame ``{date, entry_signal, exit_signal, p_win}``.
        num_bars: total number of bars.
        num_entry: number of entry signals.
        num_exit: number of exit signals.
        num_warmup_skips: bars skipped due to warm-up.
        num_ml_filtered: signals dropped by the P(win) threshold.
        elapsed_ms: total run time in milliseconds.

    """

    signals: pl.DataFrame
    num_bars: int = 0
    num_entry: int = 0
    num_exit: int = 0
    num_warmup_skips: int = 0
    num_ml_filtered: int = 0
    elapsed_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable run summary."""
        return (
            f"bars={self.num_bars} entry={self.num_entry} "
            f"exit={self.num_exit} warmup_skips={self.num_warmup_skips} "
            f"ml_filtered={self.num_ml_filtered} "
            f"time={self.elapsed_ms:.0f}ms"
        )


class _CachedContext(Context):
    """Context with a validation cache keyed by (indicator, params, attributes).

    ``Context._validate`` is deterministic for this key, but it runs on
    every indicator access on every bar -- with 5000 bars that means
    thousands of repeated manifest-validator passes. The cache is safe:
    the provider manifest does not change during a run.

    """

    def __init__(self, providers: list) -> None:
        super().__init__(providers)
        self._validation_cache: dict[tuple, bool] = {}

    def _validate(self, indicator, params, attributes):
        key = (
            indicator,
            tuple(sorted(params.items())),
            tuple(attributes),
        )
        if key not in self._validation_cache:
            super()._validate(indicator, params, attributes)
            self._validation_cache[key] = True


def _contains_let(node) -> bool:
    """Whether the AST contains let-bindings (they affect interpreter
    reuse: _locals lives on the instance - TZ-01 item 3).
    """
    import dataclasses

    from dsl.ast import Let

    if isinstance(node, Let):
        return True
    if dataclasses.is_dataclass(node):
        return any(
            _contains_let(getattr(node, f.name))
            for f in dataclasses.fields(node)
        )
    if isinstance(node, (list, tuple)):
        return any(_contains_let(c) for c in node)
    return False


def run_inference(
    df: pl.DataFrame,
    dsl_entry: str,
    dsl_exit: str | None = None,
    p_threshold: float | None = None,
    predictor=None,
) -> InferenceResult:
    """Run the strategy bar by bar and return signal series.

    Args:
        df: normalized OHLCV frame (see ``infer.data.normalize``).
        dsl_entry: entry DSL expression (required).
        dsl_exit: exit DSL expression (optional).
        p_threshold: P(win) threshold; entries below it are dropped.
        predictor: :class:`ai.src.bundle.EntryExitPredictor` or None.

    Returns:
        :class:`InferenceResult`.

    Raises:
        dsl.exceptions.ParseError: syntax error in the expressions
            (checked before the run).

    """
    started = _time.perf_counter()
    entry_ast = parse(dsl_entry)
    exit_ast = parse(dsl_exit) if dsl_exit else None

    provider = BarSeriesProvider(df)
    context = _CachedContext([provider])

    dates = df["date"]
    n = len(df)
    entry_flags = [False] * n
    exit_flags = [False] * n
    p_wins: list[float | None] = [None] * n
    warmup = 0
    ml_filtered = 0
    errors: list[str] = []

    # Interpreter reuse is safe only without let-bindings
    # (_locals lives on the instance - TZ-01 item 3).
    reuse_entry = not _contains_let(entry_ast)
    reuse_exit = exit_ast is not None and not _contains_let(exit_ast)
    entry_interp = Interpreter(context) if reuse_entry else None
    exit_interp = Interpreter(context) if reuse_exit else None

    for t in range(n):
        provider.cursor = t
        try:
            interp = entry_interp or Interpreter(context)
            entry_flags[t] = bool(interp.visit(entry_ast))
            if exit_ast is not None:
                interp = exit_interp or Interpreter(context)
                exit_flags[t] = bool(interp.visit(exit_ast))
        except WarmupNotReady:
            warmup += 1
            continue
        except ProviderError as exc:
            errors.append(f"bar {t}: provider: {exc}")
            continue

        if (
            entry_flags[t]
            and predictor is not None
            and p_threshold is not None
        ):
            p_win = predict_p_win_at(predictor, df, t)
            p_wins[t] = p_win
            if p_win is not None and p_win < p_threshold:
                entry_flags[t] = False
                ml_filtered += 1

    signals = pl.DataFrame({
        "date": dates,
        "entry_signal": entry_flags,
        "exit_signal": exit_flags,
        "p_win": [
            float(v) if v is not None else float("nan") for v in p_wins
        ],
    })
    elapsed = (_time.perf_counter() - started) * 1000.0
    return InferenceResult(
        signals=signals,
        num_bars=n,
        num_entry=int(sum(entry_flags)),
        num_exit=int(sum(exit_flags)),
        num_warmup_skips=warmup,
        num_ml_filtered=ml_filtered,
        elapsed_ms=elapsed,
        errors=errors,
    )


def _import_ta():
    """Lazy import of ta (heavy numba/talib modules)."""
    import ta.src as ta_mod

    return ta_mod


def predict_p_win_at(predictor, df: pl.DataFrame, t: int) -> float | None:
    """P(win) at bar t via the TZ-06 contract (EntryExitPredictor).

    The window is built from available columns: prices = OHLCV (5 columns),
    indicators/signals/tp/sl -- zeros (the full feature pipeline arrives in
    TZ-02/TZ-04; dimensions are cross-checked against the bundle).

    Returns:
        P(win) in [0, 1], or None if there are fewer bars than seq_len.

    """
    import torch

    seq_len = predictor.bundle.seq_len
    if t + 1 < seq_len:
        return None
    window = df[t + 1 - seq_len: t + 1]

    canon = ["open", "high", "low", "close", "volume"]
    series = {}
    for c in canon:
        s = _canon_col(window, c)
        if s is None:
            raise DSLError(f"column for {c!r} not found")
        series[c] = s.to_numpy()
    prices = torch.tensor(
        [[float(series[c][i]) for c in canon] for i in range(seq_len)],
        dtype=torch.float32,
    )
    cfg = predictor.bundle.model_config
    n_price = int(cfg.get("n_price_feats", prices.shape[1]))
    if n_price != prices.shape[1]:
        raise DSLError(
            f"price feature count mismatch: bundle expects {n_price}, "
            f"constructed {prices.shape[1]}"
        )
    n_ind = int(cfg.get("n_ind_feats", 0))
    n_sig = int(cfg.get("n_sig_feats", 0))
    n_tpsl = int(cfg.get("n_tp_sl_feats", 0))
    probs = predictor.predict_proba(
        prices,
        torch.zeros(seq_len, n_ind) if n_ind else prices[:, :0],
        torch.zeros(seq_len, n_sig) if n_sig else prices[:, :0],
        torch.zeros(seq_len, n_tpsl) if n_tpsl else prices[:, :0],
        torch.zeros(seq_len, n_tpsl) if n_tpsl else prices[:, :0],
        [],
    )
    return probs["p_win"]


def _canon_col(df: pl.DataFrame, canon: str):
    """Find a column by canonical name (with legacy-schema aliases)."""
    from .provider import _COLUMN_ALIASES

    for alias in _COLUMN_ALIASES[canon]:
        if alias in df.columns:
            return df[alias]
    return None
