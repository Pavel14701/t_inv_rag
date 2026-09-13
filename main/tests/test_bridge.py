"""Tests for the FastStream RabbitMQ bridge (TZ-09 волна 2)."""

from __future__ import annotations

import pytest

from contracts import (
    BacktestCommand,
    Candle,
    OhlcvBatch,
    QueueName,
    ReportEvent,
    SignalEvent,
    can_publish,
)
from main.src.api import WhiteAPI
from main.src.bridge import (
    HeartbeatMonitor,
    LocalBridge,
    ReconnectPolicy,
    WhiteBridge,
    report_from_payload,
)


def _batch(n: int = 2) -> OhlcvBatch:
    candles = [
        Candle(
            inst_id="BTC-USDT",
            ts=1000 + i,
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            volume=10.0,
        )
        for i in range(n)
    ]
    return OhlcvBatch(inst_id="BTC-USDT", candles=candles)


class TestReconnectPolicy:
    def test_exponential_growth(self) -> None:
        policy = ReconnectPolicy(base_delay=0.5, factor=2.0)
        assert policy.delay_for(0) == 0.5
        assert policy.delay_for(1) == 1.0
        assert policy.delay_for(2) == 2.0

    def test_capped_at_max(self) -> None:
        policy = ReconnectPolicy(base_delay=1.0, max_delay=4.0)
        assert policy.delay_for(10) == 4.0


class TestHeartbeatMonitor:
    def test_record_and_last_seen(self) -> None:
        mon = HeartbeatMonitor()
        assert mon.last_seen("q") is None
        mon.record("q", ts=100.0)
        assert mon.last_seen("q") == 100.0

    def test_stale_queues(self) -> None:
        mon = HeartbeatMonitor()
        mon.record("old", ts=0.0)
        assert "old" in mon.stale_queues(max_age_s=1.0)


class TestACL:
    def test_white_cannot_publish_evt(self) -> None:
        assert not can_publish("white", QueueName.EVT_REPORT)

    def test_local_cannot_publish_cmd(self) -> None:
        assert not can_publish("local", QueueName.CMD_BACKTEST)

    def test_white_bridge_publish_acl_enforced(self) -> None:
        from faststream.rabbit import RabbitBroker

        bridge = WhiteBridge(WhiteAPI(), broker=RabbitBroker())
        bridge._publish_queue_check(QueueName.CMD_BACKTEST)  # ok
        with pytest.raises(PermissionError):
            bridge._publish_queue_check(QueueName.EVT_REPORT)


def _make_report(request_id: str) -> ReportEvent:
    return report_from_payload(
        request_id, "completed", {"sharpe": 1.5, "pf": 2.0}
    )


class TestBridgeRoundTrip:
    @pytest.mark.asyncio
    async def test_full_backtest_flow(self) -> None:
        """cmd.backtest -> local mock -> evt.report -> WhiteAPI store."""
        from faststream.rabbit import RabbitBroker, TestRabbitBroker

        broker = RabbitBroker()
        api = WhiteAPI()
        white = WhiteBridge(api, broker=broker)
        got_cmds: list[BacktestCommand] = []

        async def on_backtest(cmd: BacktestCommand) -> ReportEvent:  # noqa: RUF029
            got_cmds.append(cmd)
            return _make_report(cmd.request_id)

        local = LocalBridge(broker=broker, on_backtest=on_backtest)
        cmd = BacktestCommand(
            request_id="job-1",
            strategy_id="s1",
            dsl_entry="rsi.value < 30",
        )
        async with TestRabbitBroker(broker):
            await white.publish_backtest(cmd)
            # local node consumed the command:
            assert len(got_cmds) == 1
            assert got_cmds[0].dsl_entry == "rsi.value < 30"
            # white registered the job when accepting the request:
            job = api.jobs.create(cmd)
            assert job.status == "pending"
            # ... local computes and reports (not nested in a handler):
            report = await on_backtest(cmd)
            await local.publish_report(report)
        status = api.get_backtest_status("job-1")
        assert status is not None
        assert status["status"] == "completed"
        assert status["report"]["sharpe"] == 1.5

    @pytest.mark.asyncio
    async def test_ohlcv_idempotent_ingest(self) -> None:
        """Duplicate md.ohlcv delivery does not duplicate candles."""
        from faststream.rabbit import RabbitBroker, TestRabbitBroker
        from msgspec import json as msgspec_json

        broker = RabbitBroker()
        got: list[OhlcvBatch] = []

        async def on_candles(batch: OhlcvBatch) -> None:  # noqa: RUF029
            got.append(batch)

        local = LocalBridge(broker=broker, on_candles=on_candles)
        batch = _batch(n=3)
        async with TestRabbitBroker(broker):
            await local.broker.publish(
                msgspec_json.encode(batch),
                queue=QueueName.MD_OHLCV.value,
            )
            await local.broker.publish(
                msgspec_json.encode(batch),
                queue=QueueName.MD_OHLCV.value,
            )
        assert len(got) == 2  # delivered twice
        # consumer-side idempotency lives in CandleStore:
        from main.src.api import CandleStore

        store = CandleStore()
        assert store.ingest_batch(batch) == 3
        assert store.ingest_batch(batch) == 0  # all duplicates

    @pytest.mark.asyncio
    async def test_signal_flow(self) -> None:
        """evt.signals -> WhiteAPI SignalStore."""
        from faststream.rabbit import RabbitBroker, TestRabbitBroker

        broker = RabbitBroker()
        api = WhiteAPI()
        WhiteBridge(api, broker=broker)
        local = LocalBridge(broker=broker)
        signal = SignalEvent(
            inst_id="BTC-USDT",
            ts=1,
            direction="long",
            entry_price=100.0,
            sl_price=95.0,
            tp_price=110.0,
            p_win=0.62,
        )
        async with TestRabbitBroker(broker):
            await local.publish_signal(signal)
        signals = api.get_signals("BTC-USDT")
        assert len(signals) == 1
        assert signals[0]["p_win"] == 0.62

    @pytest.mark.asyncio
    async def test_schema_version_tolerance(self) -> None:
        """Unsupported future schema_version is dropped, not crashed."""
        from faststream.rabbit import RabbitBroker, TestRabbitBroker
        from msgspec import json as msgspec_json

        broker = RabbitBroker()
        got: list[OhlcvBatch] = []

        async def on_candles(batch: OhlcvBatch) -> None:  # noqa: RUF029
            got.append(batch)

        local = LocalBridge(broker=broker, on_candles=on_candles)
        future = {
            "inst_id": "BTC-USDT",
            "candles": [],
            "schema_version": 99,
        }
        async with TestRabbitBroker(broker):
            await local.broker.publish(
                msgspec_json.encode(future),
                queue=QueueName.MD_OHLCV.value,
            )
        assert not got
