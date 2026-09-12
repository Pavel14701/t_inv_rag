"""Tests for risk config loading and strict validation (TZ-11 item 6)."""

from __future__ import annotations

import pytest

from risk.config import ConfigValidationError, load_config

from .helpers import build


class TestValidation:
    def test_accepts_valid_config(self) -> None:
        cfg = load_config(data=build())
        assert cfg.portfolio.max_capital_pct == pytest.approx(0.1)
        assert [r.name for r in cfg.engine.rules] == [
            "require_stop_loss",
            "position_limit",
            "max_positions",
            "daily_loss_limit",
            "drawdown_stop",
        ]

    def test_unknown_top_level_key_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown key"):
            load_config(data={"engine": {"rules": []}, "nope": 1})

    def test_unknown_rule_key_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown key"):
            load_config(
                data={"engine": {"rules": [{"name": "x", "typo": True}]}}
            )

    def test_unknown_rule_name_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown rule name"):
            load_config(
                data={"engine": {"rules": [{"name": "does_not_exist"}]}}
            )

    def test_severity_warn_rejected_in_v1(self) -> None:
        with pytest.raises(ConfigValidationError, match="only 'enforce'"):
            load_config(
                data={
                    "engine": {
                        "rules": [
                            {"name": "max_positions", "severity": "warn"}
                        ]
                    }
                }
            )

    def test_bad_capital_pct_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="must be in"):
            load_config(data=build({"portfolio": {"max_capital_pct": 1.5}}))

    def test_unknown_env_var_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown RISK"):
            load_config(data={"engine": {"rules": []}}, env={"RISK_TYPO": "1"})

    def test_env_override_applies(self) -> None:
        cfg = load_config(data=build(), env={"RISK_MAX_CAPITAL_PCT": "0.03"})
        assert cfg.portfolio.max_capital_pct == pytest.approx(0.03)

    def test_bad_env_float_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="must be a float"):
            load_config(
                data={"engine": {"rules": []}},
                env={"RISK_MAX_CAPITAL_PCT": "abc"},
            )

    def test_default_file_loads(self) -> None:
        cfg = load_config()
        assert len(cfg.engine.rules) == 5
        assert cfg.portfolio.max_capital_pct == pytest.approx(0.1)
