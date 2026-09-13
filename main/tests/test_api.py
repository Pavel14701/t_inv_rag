"""Tests for white API: idempotent ingest, jobs, signals, ACL (TZ-10)."""

from __future__ import annotations

import msgspec

from msgspec import json as msgspec_json

from contracts import (
    BacktestCommand,
    Candle,
    OhlcvBatch,
    QueueName,
    ReportEvent,
    SignalEvent,
    can_publish,
)
from main.src.api import CandleStore, JobStore, SignalStore, WhiteAPI


class TestCandleStore:
    def test_ingest_new_candle(self) -> None:
        store = CandleStore()
        c = Candle(
            inst_id="BTC", ts=1, open=1, high=2, low=0.5, close=1.5, volume=10
        )
        assert store.ingest(c) is True

    def test_ingest_duplicate_ignored(self) -> None:
        store = CandleStore()
        c = Candle(
            inst_id="BTC", ts=1, open=1, high=2, low=0.5, close=1.5, volume=10
        )
        store.ingest(c)
        assert store.ingest(c) is False  # idempotent
        assert store.count() == 1

    def test_batch_ingest_counts_new(self) -> None:
        store = CandleStore()
        batch = OhlcvBatch(
            inst_id="BTC",
            candles=[
                Candle(
                    inst_id="BTC",
                    ts=1,
                    open=1,
                    high=2,
                    low=0.5,
                    close=1.5,
                    volume=10,
                ),
                Candle(
                    inst_id="BTC",
                    ts=2,
                    open=1.5,
                    high=2.5,
                    low=1,
                    close=2,
                    volume=20,
                ),
            ],
        )
        assert store.ingest_batch(batch) == 2
        assert store.ingest_batch(batch) == 0  # all duplicates


class TestJobStore:
    def test_create_and_get(self) -> None:
        store = JobStore()
        cmd = BacktestCommand(
            request_id="j1", strategy_id="s1", dsl_entry="rsi < 30"
        )
        job = store.create(cmd)
        assert job.status == "pending"
        assert store.get("j1") is not None

    def test_lifecycle(self) -> None:
        store = JobStore()
        cmd = BacktestCommand(
            request_id="j2", strategy_id="s1", dsl_entry="close > open"
        )
        store.create(cmd)
        report = ReportEvent(
            request_id="j2", report_json='{"pf": 1.5}', status="completed"
        )
        store.set_report("j2", report)
        job = store.get("j2")
        assert job is not None
        assert job.status == "completed"
        assert job.report is not None


class TestSignalStore:
    def test_add_and_latest(self) -> None:
        store = SignalStore()
        sig = SignalEvent(
            inst_id="BTC",
            ts=1,
            direction="long",
            entry_price=100,
            sl_price=95,
            tp_price=110,
        )
        store.add(sig)
        latest = store.latest("BTC")
        assert len(latest) == 1
        assert latest[0].direction == "long"

    def test_max_per_inst(self) -> None:
        store = SignalStore(max_per_inst=2)
        for ts in range(5):
            store.add(
                SignalEvent(
                    inst_id="BTC",
                    ts=ts,
                    direction="long",
                    entry_price=100,
                    sl_price=95,
                    tp_price=110,
                )
            )
        assert len(store.latest("BTC")) == 2


class TestWhiteAPI:
    def test_handle_report_updates_job(self) -> None:
        api = WhiteAPI()
        cmd = BacktestCommand(
            request_id="r1", strategy_id="s1", dsl_entry="close > open"
        )
        api.submit_backtest(cmd)
        report = ReportEvent(
            request_id="r1", report_json='{"pf": 1.5}', status="completed"
        )
        api.handle_report(msgspec.json.encode(report))
        status = api.get_backtest_status("r1")
        assert status is not None
        assert status["status"] == "completed"
        assert status["report"]["pf"] == 1.5

    def test_handle_signal_adds_to_store(self) -> None:
        api = WhiteAPI()
        sig = SignalEvent(
            inst_id="BTC",
            ts=1,
            direction="long",
            entry_price=100,
            sl_price=95,
            tp_price=110,
        )
        api.handle_signal(msgspec_json.encode(sig))
        signals = api.get_signals("BTC")
        assert len(signals) == 1

    def test_acl_blocks_local_publish_to_cmd(self) -> None:
        assert not can_publish("local", QueueName.CMD_BACKTEST)
        assert not can_publish("local", QueueName.CMD_TRAIN)

    def test_full_backtest_flow(self) -> None:
        """POST /backtests -> cmd -> evt.report -> GET returns report."""
        api = WhiteAPI()
        cmd = BacktestCommand(
            request_id="job-42",
            strategy_id="siv",
            dsl_entry="rsi.value < 30",
        )
        job = api.submit_backtest(cmd)
        assert job.job_id == "job-42"
        assert job.status == "pending"
        # Simulate local node response
        report = ReportEvent(
            request_id="job-42",
            report_json='{"profit_factor": 2.1, "sharpe": 1.3}',
            status="completed",
        )
        api.handle_report(msgspec_json.encode(report))
        status = api.get_backtest_status("job-42")
        assert status is not None
        assert status["status"] == "completed"
        assert status["report"]["profit_factor"] == 2.1

    def test_get_backtest_status_not_found(self) -> None:
        api = WhiteAPI()
        assert api.get_backtest_status("nonexistent") is None
