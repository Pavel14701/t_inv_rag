"""Risk gate integration in the backtest engine (TZ-11 item 4.5/7).

Acceptance checks: config-driven behavior (two configs, one code ->
different results), reject audit trail (rule name + params snapshot +
counter), and the permissive-config equivalence invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from backtest.contracts import BacktestConfig
from backtest.engine import run_backtest
from risk.config import ConfigValidationError, RiskConfig, load_config


@pytest.fixture
def ohlc_arrays() -> dict[str, np.ndarray]:
    """Synthetic OHLCV + ATR for 500 bars (same shape as TZ-04 tests)."""
    rng = np.random.default_rng(42)
    n = 500
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    atr = np.convolve(tr, np.ones(14) / 14, mode="same")
    atr[:14] = np.nan
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "atr": atr,
    }


def _entry_signals(n: int = 500, every: int = 50) -> np.ndarray:
    sig = np.zeros(n, dtype=bool)
    sig[every::every] = True
    return sig


def _exit_signals(n: int = 500) -> np.ndarray:
    sig = np.zeros(n, dtype=bool)
    sig[10::50] = True  # force exits so new entries can happen
    return sig


def _run(arrs, risk_config):
    return run_backtest(
        arrs["open"],
        arrs["high"],
        arrs["low"],
        arrs["close"],
        arrs["atr"],
        _entry_signals(),
        _exit_signals(),
        BacktestConfig(),
        risk_config=risk_config,
    )


class TestConfigDrivenBehavior:
    """TZ-11 acceptance: two configs, one code, different behavior."""

    def test_permissive_config_allows_trades(self, ohlc_arrays) -> None:
        cfg = load_config(
            data={
                "engine": {
                    "rules": [
                        {
                            "name": "position_limit",
                            "active": True,
                            "params": {"max_capital_pct": 1.0},
                        }
                    ]
                }
            }
        )
        _, _, metrics = _run(ohlc_arrays, cfg)
        assert metrics.n_trades >= 1
        assert len(metrics.risk_rejects) == 0

    def test_strict_config_rejects_everything(self, ohlc_arrays) -> None:
        cfg = load_config(
            data={
                "engine": {
                    "rules": [
                        {
                            "name": "require_stop_loss",
                            "active": True,
                            "params": {
                                "sl": {"min_mult": 0.0},
                                "tp": {"min_mult": 100.0},
                            },
                        }
                    ]
                }
            }
        )
        _, _, metrics = _run(ohlc_arrays, cfg)
        assert metrics.n_trades == 0
        assert len(metrics.risk_rejects) == 9  # one per entry signal
        assert all(r.rule == "require_stop_loss" for r in metrics.risk_rejects)

    def test_permissive_matches_no_risk_baseline(self, ohlc_arrays) -> None:
        """Permissive gate must not distort the TZ-04 baseline result."""
        cfg = load_config(
            data={
                "engine": {
                    "rules": [
                        {
                            "name": "position_limit",
                            "active": True,
                            "params": {"max_capital_pct": 1.0},
                        }
                    ]
                }
            }
        )
        baseline_trades, baseline_eq, baseline_metrics = run_backtest(
            ohlc_arrays["open"],
            ohlc_arrays["high"],
            ohlc_arrays["low"],
            ohlc_arrays["close"],
            ohlc_arrays["atr"],
            _entry_signals(),
            _exit_signals(),
            BacktestConfig(),
        )
        gated_trades, gated_eq, gated_metrics = _run(ohlc_arrays, cfg)
        assert len(gated_trades) == len(baseline_trades)
        assert np.allclose(gated_eq, baseline_eq)
        assert gated_metrics.total_pnl == pytest.approx(
            baseline_metrics.total_pnl
        )


class TestRejectAuditTrail:
    """TZ-11 item 7: rejects are stored with name, params, counter."""

    def test_reject_record_carries_params_snapshot(self, ohlc_arrays) -> None:
        cfg = load_config(
            data={
                "engine": {
                    "rules": [
                        {
                            "name": "daily_loss_limit",
                            "active": True,
                            "params": {"max_daily_loss_pct": 0.0},
                        }
                    ]
                }
            }
        )
        _, _, metrics = _run(ohlc_arrays, cfg)
        assert len(metrics.risk_rejects) >= 1
        first = metrics.risk_rejects[0]
        assert first.rule == "daily_loss_limit"
        assert first.params == {"max_daily_loss_pct": 0.0}
        assert "daily_loss_limit" in first.reason
        assert first.bar_idx >= 0

    def test_reject_counter_equals_recorded_rejects(self, ohlc_arrays) -> None:
        cfg = load_config(
            data={
                "engine": {
                    "rules": [
                        {
                            "name": "require_stop_loss",
                            "active": True,
                            "params": {
                                "sl": {"min_mult": 0.0},
                                "tp": {"min_mult": 100.0},
                            },
                        }
                    ]
                }
            }
        )
        trades, _, metrics = _run(ohlc_arrays, cfg)
        assert len(trades) == 0
        assert len(metrics.risk_rejects) == 9


class TestDefaultConfigIntegration:
    """The shipped configs/risk.yaml drives the engine end-to-end."""

    def test_default_yaml_runs(self, ohlc_arrays) -> None:
        cfg: RiskConfig = load_config()  # configs/risk.yaml from the repo
        trades, _, metrics = _run(ohlc_arrays, cfg)
        assert metrics.n_trades == len(trades)
        # Every reject (if any - synthetic ATR shrinks relative to the
        # random-walk price) carries a full audit record.
        for reject in metrics.risk_rejects:
            assert reject.rule == "require_stop_loss"
            assert reject.params["tp"]["min_mult"] == 1.0


class TestParamsValidation:
    """Unknown/wrong-typed rule params raise at validate_config time."""

    def test_unknown_param_key_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown param"):
            load_config(
                data={
                    "engine": {
                        "rules": [
                            {
                                "name": "max_positions",
                                "active": True,
                                "params": {"max_open_typo": 5},
                            }
                        ]
                    }
                }
            )

    def test_wrong_param_type_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="must be"):
            load_config(
                data={
                    "engine": {
                        "rules": [
                            {
                                "name": "max_positions",
                                "active": True,
                                "params": {"max_open": "five"},
                            }
                        ]
                    }
                }
            )

    def test_bool_is_not_a_number(self) -> None:
        with pytest.raises(ConfigValidationError, match="must be"):
            load_config(
                data={
                    "engine": {
                        "rules": [
                            {
                                "name": "max_positions",
                                "active": True,
                                "params": {"max_open": True},
                            }
                        ]
                    }
                }
            )

    def test_nested_missing_key_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="required"):
            load_config(
                data={
                    "engine": {
                        "rules": [
                            {
                                "name": "require_stop_loss",
                                "active": True,
                                "params": {"tp": {"min_mult": 1.0}},
                            }
                        ]
                    }
                }
            )

    def test_nested_unknown_key_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown param"):
            load_config(
                data={
                    "engine": {
                        "rules": [
                            {
                                "name": "require_stop_loss",
                                "active": True,
                                "params": {
                                    "sl": {
                                        "min_mult": 0.0,
                                        "use_atr": True,
                                    },
                                    "tp": {"min_mult": 1.0},
                                },
                            }
                        ]
                    }
                }
            )

    def test_valid_params_pass(self) -> None:
        cfg = load_config(
            data={
                "engine": {
                    "rules": [
                        {
                            "name": "require_stop_loss",
                            "active": True,
                            "params": {
                                "sl": {"min_mult": 0.0},
                                "tp": {"min_mult": 1.0},
                            },
                        },
                        {
                            "name": "max_positions",
                            "active": True,
                            "params": {"max_open": 3},
                        },
                    ]
                }
            }
        )
        assert len(cfg.engine.rules) == 2
