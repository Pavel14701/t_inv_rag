"""Tests for temporal validation, walk-forward, and
baseline gate (TZ-04 п.4.6)."""

from __future__ import annotations

from backtest.validation import (
    BaselineGateStatus,
    baseline_gate,
    temporal_split,
    validate_report_has_baselines,
    walk_forward_folds,
)


class TestTemporalSplit:
    def test_70_30_split(self) -> None:
        train_size, test_size = temporal_split(1000, train_fraction=0.7)
        assert train_size == 700
        assert test_size == 300
        assert train_size + test_size == 1000

    def test_no_overlap(self) -> None:
        train_size, test_size = temporal_split(100, train_fraction=0.8)
        assert train_size == 80
        assert test_size == 20


class TestWalkForward:
    def test_basic_folds(self) -> None:
        folds = walk_forward_folds(1000, train_window=500, test_window=100)
        assert len(folds) >= 1
        for start, train_end, test_start, test_end in folds:
            assert train_end == start + 500
            assert test_start == train_end
            assert test_end == test_start + 100

    def test_folds_are_chronological(self) -> None:
        folds = walk_forward_folds(1000, 300, 100)
        for i in range(1, len(folds)):
            assert folds[i][0] > folds[i - 1][0]

    def test_no_lookahead_in_folds(self) -> None:
        """Training window must never overlap test window."""
        folds = walk_forward_folds(1000, 500, 100)
        for _start, train_end, test_start, _ in folds:
            assert train_end <= test_start


class TestBaselineGate:
    def test_pass_when_transformer_better(self) -> None:
        result = baseline_gate(
            transformer_sharpe=1.5,
            transformer_pf=2.0,
            baseline_sharpes={"lr": 1.0, "rf": 1.1, "xgb": 1.2, "bh": 0.3},
            baseline_pfs={"lr": 1.5, "rf": 1.6, "xgb": 1.7, "bh": 1.0},
            min_improvement_pct=5.0,
        )
        assert result.status == BaselineGateStatus.PASS

    def test_fail_when_transformer_worse(self) -> None:
        result = baseline_gate(
            transformer_sharpe=0.5,
            transformer_pf=1.2,
            baseline_sharpes={"lr": 1.0, "rf": 1.1},
            baseline_pfs={"lr": 1.5, "rf": 1.6},
        )
        assert result.status == BaselineGateStatus.FAIL

    def test_simplify_when_equal(self) -> None:
        """Not worse but not >5% better -> simplify."""
        result = baseline_gate(
            transformer_sharpe=1.2,
            transformer_pf=1.8,
            baseline_sharpes={"lr": 1.0, "rf": 1.2},
            baseline_pfs={"lr": 1.5, "rf": 1.8},
            min_improvement_pct=5.0,
        )
        assert result.status == BaselineGateStatus.SIMPLIFY

    def test_empty_baselines_fail(self) -> None:
        result = baseline_gate(
            transformer_sharpe=1.0,
            transformer_pf=1.5,
            baseline_sharpes={},
            baseline_pfs={},
        )
        assert result.status == BaselineGateStatus.FAIL

    def test_no_baselines_rejected(self) -> None:
        errors = validate_report_has_baselines({"n_trades": 10})
        assert len(errors) == 2  # missing baselines + missing gate

    def test_partial_baselines_rejected(self) -> None:
        errors = validate_report_has_baselines(
            {
                "baselines": {"buy_and_hold": {"sharpe": 0.3}},
            }
        )
        assert any("Missing baseline" in e for e in errors)


class TestReportValidation:
    def test_valid_report_passes(self) -> None:
        report = {
            "baselines": {
                "buy_and_hold": {"sharpe": 0.3},
                "logistic_regression": {"sharpe": 1.0},
                "random_forest": {"sharpe": 1.1},
                "xgboost": {"sharpe": 1.2},
            },
            "baseline_gate": "pass",
        }
        assert validate_report_has_baselines(report) == []

    def test_missing_baselines_rejected(self) -> None:
        report = {"n_trades": 10, "profit_factor": 1.8}
        errors = validate_report_has_baselines(report)
        assert len(errors) > 0
