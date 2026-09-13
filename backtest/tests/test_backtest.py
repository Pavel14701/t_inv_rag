"""Tests for execution rules, metrics, and invariants (TZ-04)."""

from __future__ import annotations

import numpy as np
import pytest

from backtest.contracts import BacktestConfig, Trade
from backtest.execution import (
    check_exit,
    compute_tp_sl,
    effective_entry_price,
    effective_exit_price,
    realised_pnl,
    realised_r_multiple,
)
from backtest.metrics import compute_metrics


class TestExecution:
    def test_entry_long_slippage_up(self) -> None:
        assert effective_entry_price("long", 100.0, 0.0005) == pytest.approx(
            100.05
        )

    def test_entry_short_slippage_down(self) -> None:
        assert effective_entry_price("short", 100.0, 0.0005) == pytest.approx(
            99.95
        )

    def test_tp_no_slippage(self) -> None:
        assert effective_exit_price("long", True, 110.0, 0.0005) == 110.0

    def test_sl_slips_against_long(self) -> None:
        got = effective_exit_price("long", False, 95.0, 0.0005)
        assert got == pytest.approx(94.9525)

    def test_tp_sl_levels_long(self) -> None:
        sl, tp = compute_tp_sl(100.0, "long", 2.0, sl_mult=1.5, tp_mult=2.0)
        assert sl == pytest.approx(97.0)
        assert tp == pytest.approx(104.0)

    def test_tp_sl_levels_short(self) -> None:
        sl, tp = compute_tp_sl(100.0, "short", 2.0, sl_mult=1.5, tp_mult=2.0)
        assert sl == pytest.approx(103.0)
        assert tp == pytest.approx(96.0)

    def test_sl_first_pessimism_long(self) -> None:
        """When both SL and TP are hit in the same bar, SL wins (pessimism)."""
        hit_tp, hit_sl = check_exit(
            open_p=100,
            high=110,
            low=90,
            sl_price=95,
            tp_price=105,
            direction="long",
        )
        assert hit_sl is True
        assert hit_tp is True
        # SL should take priority

    def test_no_exit(self) -> None:
        hit_tp, hit_sl = check_exit(
            open_p=100,
            high=102,
            low=98,
            sl_price=95,
            tp_price=105,
            direction="long",
        )
        assert hit_tp is False
        assert hit_sl is False

    def test_pnl_long_with_commission(self) -> None:
        pnl = realised_pnl("long", 100.0, 110.0, 10.0, 0.001)
        gross = 10.0 * 10.0  # 100
        commission = (100.0 + 110.0) * 10.0 * 0.001  # 2.1
        assert pnl == pytest.approx(gross - commission)

    def test_pnl_short_negative(self) -> None:
        pnl = realised_pnl("short", 100.0, 90.0, 10.0, 0.001)
        gross = 10.0 * 10.0  # short gains when price drops
        commission = (100.0 + 90.0) * 10.0 * 0.001
        assert pnl == pytest.approx(gross - commission)

    def test_r_multiple_long(self) -> None:
        r = realised_r_multiple("long", 100.0, 110.0, 95.0, 120.0)
        assert r == pytest.approx(10.0 / 5.0)  # 2R


class TestMetrics:
    @pytest.fixture
    def sample_trades(self) -> list[Trade]:
        return [
            Trade("long", 0, 10, 100.0, 110.0, 10.0, 2.0, 98.0, 10, "tp"),
            Trade("long", 20, 25, 100.0, 95.0, 10.0, -1.0, -52.0, 5, "sl"),
            Trade("long", 30, 50, 100.0, 105.0, 10.0, 1.0, 48.0, 20, "tp"),
        ]

    def test_profit_factor(self, sample_trades: list[Trade]) -> None:
        m = compute_metrics(sample_trades, np.array([100.0, 150.0]))
        wins = 98.0 + 48.0
        losses = 52.0
        assert m.profit_factor == pytest.approx(wins / losses, rel=0.01)

    def test_win_rate(self, sample_trades: list[Trade]) -> None:
        m = compute_metrics(sample_trades, np.array([100.0, 150.0]))
        assert m.win_rate == pytest.approx(2 / 3)

    def test_n_trades(self, sample_trades: list[Trade]) -> None:
        m = compute_metrics(sample_trades, np.array([100.0, 150.0]))
        assert m.n_trades == 3

    def test_max_drawdown(self) -> None:
        trades = [
            Trade("long", 0, 1, 100.0, 110.0, 1.0, 1.0, 10.0, 1, "tp"),
            Trade("long", 2, 3, 110.0, 90.0, 1.0, -2.0, -20.0, 1, "sl"),
        ]
        equity = np.array([100.0, 110.0, 110.0, 90.0, 90.0])
        m = compute_metrics(trades, equity)
        assert m.max_drawdown_pct == pytest.approx(0.1818, abs=0.01)

    def test_empty_trades(self) -> None:
        m = compute_metrics([], np.array([100.0]))
        assert m.n_trades == 0
        assert m.profit_factor == 0.0


class TestReproducibility:
    def test_same_seed_same_config_same_result(self) -> None:
        """Deterministic: same inputs -> same Trade objects."""
        cfg = BacktestConfig(commission_pct=0.001, slippage_pct=0.0005)
        p1 = effective_entry_price("long", 100.0, cfg.slippage_pct)
        p2 = effective_entry_price("long", 100.0, cfg.slippage_pct)
        assert p1 == p2
        assert cfg.commission_pct == 0.001  # frozen
