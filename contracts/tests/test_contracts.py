"""Tests for message contracts: serialisation, ACL, and schema (TZ-09)."""

from __future__ import annotations

import msgspec
import pytest

from contracts import (
    BacktestCommand,
    Candle,
    OhlcvBatch,
    QueueName,
    ReportEvent,
    SignalEvent,
    TrainCommand,
    can_consume,
    can_publish,
)


class TestSerialization:
    def test_candle_roundtrip(self) -> None:
        candle = Candle(
            inst_id="BTC-USDT",
            ts=1704067200000,
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42300.0,
            volume=1234.5,
        )
        data = msgspec.json.encode(candle)
        decoded = msgspec.json.decode(data, type=Candle)
        assert decoded == candle

    def test_ohlcv_batch_roundtrip(self) -> None:
        batch = OhlcvBatch(
            inst_id="BTC-USDT",
            candles=[
                Candle(
                    inst_id="BTC-USDT",
                    ts=1,
                    open=1,
                    high=2,
                    low=0.5,
                    close=1.5,
                    volume=10,
                ),
                Candle(
                    inst_id="BTC-USDT",
                    ts=2,
                    open=1.5,
                    high=2.5,
                    low=1,
                    close=2,
                    volume=20,
                ),
            ],
        )
        data = msgspec.json.encode(batch)
        decoded = msgspec.json.decode(data, type=OhlcvBatch)
        assert decoded == batch
        assert len(decoded.candles) == 2

    def test_signal_event_roundtrip(self) -> None:
        sig = SignalEvent(
            inst_id="BTC-USDT",
            ts=1704067200000,
            direction="long",
            entry_price=42300.0,
            sl_price=41800.0,
            tp_price=43500.0,
            p_win=0.72,
            strategy_id="siv-v1",
        )
        data = msgspec.json.encode(sig)
        decoded = msgspec.json.decode(data, type=SignalEvent)
        assert decoded == sig
        assert decoded.p_win == pytest.approx(0.72)

    def test_backtest_command_roundtrip(self) -> None:
        cmd = BacktestCommand(
            request_id="req-001",
            strategy_id="siv-v1",
            dsl_entry="rsi.value < 30",
            dsl_exit="close > open",
        )
        data = msgspec.json.encode(cmd)
        decoded = msgspec.json.decode(data, type=BacktestCommand)
        assert decoded == cmd

    def test_report_event_roundtrip(self) -> None:
        report = ReportEvent(
            request_id="req-001",
            report_json='{"pf": 1.8}',
            status="completed",
        )
        data = msgspec.json.encode(report)
        decoded = msgspec.json.decode(data, type=ReportEvent)
        assert decoded == report

    def test_unknown_field_ignored(self) -> None:
        """msgspec silently ignores unknown fields during decode."""
        data = b"{"
        data += b'"inst_id": "BTC", "ts": 1, "open": 1, "high": 1,'
        data += b'"low": 1, "close": 1, "volume": 1, "u": 1}'
        candle = msgspec.json.decode(data, type=Candle)
        assert candle.inst_id == "BTC"
        assert candle.inst_id == "BTC"

    def test_schema_version_default(self) -> None:
        candle = Candle(
            inst_id="BTC", ts=1, open=1, high=1, low=1, close=1, volume=1
        )
        assert candle.schema_version == 1


class TestACL:
    """ACL: white cannot publish to evt.*, local cannot publish to cmd.*."""

    def test_white_can_publish_data(self) -> None:
        assert can_publish("white", QueueName.MD_OHLCV)
        assert can_publish("white", QueueName.CMD_BACKTEST)

    def test_white_cannot_publish_events(self) -> None:
        assert not can_publish("white", QueueName.EVT_REPORT)
        assert not can_publish("white", QueueName.EVT_SIGNALS)

    def test_local_can_publish_events(self) -> None:
        assert can_publish("local", QueueName.EVT_REPORT)
        assert can_publish("local", QueueName.EVT_SIGNALS)

    def test_local_cannot_publish_commands(self) -> None:
        assert not can_publish("local", QueueName.CMD_BACKTEST)
        assert not can_publish("local", QueueName.CMD_TRAIN)

    def test_local_can_consume_data(self) -> None:
        assert can_consume("local", QueueName.MD_OHLCV)
        assert can_consume("local", QueueName.CMD_BACKTEST)

    def test_white_can_consume_events(self) -> None:
        assert can_consume("white", QueueName.EVT_REPORT)
        assert can_consume("white", QueueName.EVT_SIGNALS)

    def test_no_risk_limit_change_command(self) -> None:
        """TZ-11 invariant: no command to change risk limits exists."""
        cmd_fields = set(BacktestCommand.__struct_fields__)
        train_fields = set(TrainCommand.__struct_fields__)
        assert "risk_limits" not in cmd_fields
        assert "risk_limits" not in train_fields
        assert "update_risk" not in cmd_fields
