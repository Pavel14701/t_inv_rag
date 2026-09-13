"""Tests for Strategy validation and AST indicator extraction (TZ-02 п.3)."""

from __future__ import annotations

from strategies.src.application.strategy import (
    Strategy,
    indicators_used,
    validate_strategy,
)


class TestValidateStrategy:
    def test_valid_entry_only(self, etalon: Strategy, manifest: dict) -> None:
        assert validate_strategy(etalon, manifest) == []

    def test_valid_entry_and_exit(
        self, etalon: Strategy, manifest: dict
    ) -> None:
        s = Strategy(
            id="x",
            name="x",
            description="x",
            dsl_entry="rsi.value < 30",
            dsl_exit="close > open",
        )
        assert validate_strategy(s, manifest) == []

    def test_unknown_indicator_rejected(self, manifest: dict) -> None:
        s = Strategy(
            id="x", name="x", description="x", dsl_entry="stoch_rsi < 20"
        )
        errors = validate_strategy(s, manifest)
        assert len(errors) == 1
        assert "Unknown indicator" in errors[0]
        assert "stoch_rsi" in errors[0]

    def test_parse_error_reported(self, manifest: dict) -> None:
        s = Strategy(id="x", name="x", description="x", dsl_entry="close >")
        errors = validate_strategy(s, manifest)
        assert len(errors) == 1
        assert "parse error" in errors[0].lower()


class TestIndicatorsUsed:
    def test_single_indicator(self) -> None:
        assert indicators_used("rsi.value < 30") == ["rsi"]

    def test_multiple_indicators(self) -> None:
        result = indicators_used("rsi.value < 30 and rising(close, 5)")
        assert "rsi" in result
        assert "close" in result

    def test_close_and_open_are_indicators(self) -> None:
        result = indicators_used("close > open")
        assert "close" in result
        assert "open" in result
