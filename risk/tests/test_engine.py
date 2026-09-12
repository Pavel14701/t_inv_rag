"""Tests for the config-driven risk engine: invariants, edge cases,
config-driven behavior and per-instrument overrides (TZ-11 item 6)."""

from __future__ import annotations

import pytest

from risk.config import load_config
from risk.engine import PortfolioState, Signal, check

from .helpers import build


def _cfg(**kw) -> object:
    return load_config(data=build(kw or None))


class TestInvariants:
    def test_approve_only_when_all_rules_pass(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        decision = check(entry, portfolio, _cfg())
        assert decision.approve is True
        assert decision.size > 0

    def test_no_approve_without_sl(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        no_sl = Signal(
            instrument=entry.instrument,
            entry_price=entry.entry_price,
            requested_units=entry.requested_units,
            has_sl=False,
            has_tp=True,
            tp_price=120.0,
        )
        decision = check(no_sl, portfolio, _cfg())
        assert decision.approve is False
        assert "require_stop_loss" in decision.reason

    def test_size_zero_on_empty_capital(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        portfolio.capital = 0.0
        decision = check(entry, portfolio, _cfg())
        assert decision.approve is False
        assert decision.size == pytest.approx(0.0)

    def test_reject_is_final_no_partial_approve(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        portfolio.open_positions = 100  # far above max_open
        decision = check(entry, portfolio, _cfg())
        assert decision.approve is False
        assert decision.size == pytest.approx(0.0)


class TestEdgeCases:
    def test_daily_loss_limit_blocks_rest_of_day(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        portfolio.day_pnl = -3000.0  # over 2% of 100k
        decision = check(entry, portfolio, _cfg())
        assert decision.approve is False
        assert "daily_loss_limit" in decision.reason

    def test_drawdown_stop_blocks(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        portfolio.capital = 70_000.0  # 30% below peak
        portfolio.peak_capital = 100_000.0
        decision = check(entry, portfolio, _cfg())
        assert decision.approve is False
        assert "drawdown_stop" in decision.reason

    def test_max_positions_blocks(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        portfolio.open_positions = 5
        decision = check(entry, portfolio, _cfg())
        assert decision.approve is False
        assert "max_positions" in decision.reason

    def test_position_limit_caps_size(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        # capital * 10% / 100 = 100 units for a 100k capital
        decision = check(entry, portfolio, _cfg())
        assert decision.size == pytest.approx(100.0)


class TestConfigDriven:
    def test_tighter_capital_pct_reduces_size(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        cfg = _cfg(portfolio={"max_capital_pct": 0.05})
        decision = check(entry, portfolio, cfg)
        assert decision.size == pytest.approx(50.0)

    def test_disabled_rule_is_ignored(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        rules = build()["engine"]["rules"]
        rules[2] = dict(rules[2], active=False)  # max_positions off
        cfg = load_config(
            data={
                "engine": {"rules": rules},
                "portfolio": build()["portfolio"],
            }
        )
        portfolio.open_positions = 100
        decision = check(entry, portfolio, cfg)
        assert decision.approve is True

    def test_per_instrument_override_shadows_portfolio(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        cfg = load_config(
            data={
                "engine": build()["engine"],
                "portfolio": {
                    "max_capital_pct": 0.1,
                    "max_units": None,
                    "instruments": {
                        "TQBR.FIGI": {"max_capital_pct": 0.05},
                    },
                },
            }
        )
        decision = check(entry, portfolio, cfg)
        assert decision.size == pytest.approx(50.0)

    def test_two_configs_different_behavior_same_code(
        self, portfolio: PortfolioState, entry: Signal
    ) -> None:
        strict = load_config(
            data={
                "engine": {
                    "rules": [
                        dict(
                            build()["engine"]["rules"][2],
                            params={"max_open": 1},
                        )
                    ]
                },
                "portfolio": {
                    "max_capital_pct": 0.1,
                    "max_units": None,
                    "instruments": {},
                },
            }
        )
        loose = load_config(
            data={
                "engine": {
                    "rules": [
                        dict(
                            build()["engine"]["rules"][2],
                            params={"max_open": 10},
                        )
                    ]
                },
                "portfolio": {
                    "max_capital_pct": 0.1,
                    "max_units": None,
                    "instruments": {},
                },
            }
        )
        portfolio.open_positions = 3
        assert check(entry, portfolio, strict).approve is False
        assert check(entry, portfolio, loose).approve is True
