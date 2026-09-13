"""Temporal validation: OOS split, walk-forward, baseline gate (TZ-04 п.4.6).

The Baseline Gate is a blocking filter: without a passing comparison
against simple methods, results are invalid and downstream consumers
(RAG, live) are blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ValidationMode(Enum):
    """How to split train/test data."""

    TEMPORAL = "temporal"
    WALK_FORWARD = "walk_forward"


class BaselineGateStatus(Enum):
    """Result of the baseline comparison gate."""

    PASS = "pass"
    FAIL = "fail"
    SIMPLIFY = "simplify"


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    profit_factor: float
    sharpe: float


@dataclass(frozen=True, slots=True)
class BaselineGateResult:
    """Output of the mandatory baseline comparison gate (п.4.6.1)."""

    status: BaselineGateStatus
    best_baseline_sharpe: float
    transformer_sharpe: float
    sharpe_improvement_pct: float
    best_baseline_pf: float
    transformer_pf: float
    pf_improvement_pct: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregated validation results."""

    mode: ValidationMode
    folds: list[FoldResult]
    baseline_gate: BaselineGateResult | None
    is_valid: bool  # False if baseline gate failed


def temporal_split(
    n_bars: int,
    train_fraction: float = 0.7,
) -> tuple[int, int]:
    """Temporal split: train = [0, split_point), test = [split_point, n).

    No shuffling, no overlap. The most basic requirement for financial
    time-series validation.
    """
    split = int(n_bars * train_fraction)
    return split, n_bars - split


def walk_forward_folds(
    n_bars: int,
    train_window: int,
    test_window: int,
    step: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Generate walk-forward fold boundaries.

    Args:
        n_bars: total number of bars.
        train_window: bars in each training window.
        test_window: bars in each test window.
        step: slide step (defaults to test_window).

    Returns:
        List of (train_start, train_end, test_start, test_end) tuples.

    """
    if step is None:
        step = test_window
    folds = []
    start = 0
    while start + train_window + test_window <= n_bars:
        train_end = start + train_window
        test_start = train_end
        test_end = test_start + test_window
        folds.append((start, train_end, test_start, test_end))
        start += step
    return folds


# --------------------------------------------------------------------------- #
# Baseline Gate (TZ-04 п.4.6.1)
# --------------------------------------------------------------------------- #


def baseline_gate(
    transformer_sharpe: float,
    transformer_pf: float,
    baseline_sharpes: dict[str, float],
    baseline_pfs: dict[str, float],
    min_improvement_pct: float = 5.0,
) -> BaselineGateResult:
    """Evaluate the baseline gate: Transformer vs simple methods.

    Args:
        transformer_sharpe: Sharpe ratio of the Transformer model.
        transformer_pf: Profit Factor of the Transformer model.
        baseline_sharpes: {name: sharpe} for each baseline.
        baseline_pfs: {name: pf} for each baseline.
        min_improvement_pct: required improvement (>5-10%).

    Returns:
        BaselineGateResult with pass/fail/simplify status.

    """
    if not baseline_sharpes or not baseline_pfs:
        return BaselineGateResult(
            status=BaselineGateStatus.FAIL,
            best_baseline_sharpe=0.0,
            transformer_sharpe=transformer_sharpe,
            sharpe_improvement_pct=0.0,
            best_baseline_pf=0.0,
            transformer_pf=transformer_pf,
            pf_improvement_pct=0.0,
        )

    best_sharpe = max(baseline_sharpes.values())
    best_pf = max(baseline_pfs.values())

    sharpe_impr = (
        ((transformer_sharpe - best_sharpe) / abs(best_sharpe) * 100)
        if best_sharpe != 0
        else 0.0
    )
    pf_impr = (
        ((transformer_pf - best_pf) / abs(best_pf) * 100)
        if best_pf != 0
        else 0.0
    )

    if sharpe_impr > min_improvement_pct or pf_impr > min_improvement_pct:
        status = BaselineGateStatus.PASS
    elif transformer_sharpe >= best_sharpe and transformer_pf >= best_pf:
        status = BaselineGateStatus.SIMPLIFY
    else:
        status = BaselineGateStatus.FAIL

    return BaselineGateResult(
        status=status,
        best_baseline_sharpe=best_sharpe,
        transformer_sharpe=transformer_sharpe,
        sharpe_improvement_pct=sharpe_impr,
        best_baseline_pf=best_pf,
        transformer_pf=transformer_pf,
        pf_improvement_pct=pf_impr,
    )


def validate_report_has_baselines(
    report: dict,
) -> list[str]:
    """Check that a backtest report contains the mandatory baseline section.

    Returns a list of errors (empty = valid).
    """
    errors = []
    if "baselines" not in report:
        errors.append("Report missing 'baselines' section (TZ-04 п.4.6.1)")
    else:
        baselines = report["baselines"]
        required = {
            "buy_and_hold",
            "logistic_regression",
            "random_forest",
            "xgboost",
        }
        missing = required - set(baselines)
        if missing:
            errors.append(f"Missing baseline(s): {sorted(missing)}")
    if "baseline_gate" not in report:
        errors.append("Report missing 'baseline_gate' status field")
    return errors
