"""Service-composition tests: bridges wired to PostgreSQL stores (TZ-10).

SQLite plays the PG role; the flow (TestRabbitBroker -> handler -> PG)
is identical for PostgreSQL. Phase-split like test_bridge.py: publishes
cannot be nested inside subscriber handlers under TestRabbitBroker.
"""

from __future__ import annotations

import pytest

from msgspec import json as msgspec_json

from contracts import (
    BacktestCommand,
    Candle,
    OhlcvBatch,
    QueueName,
)
from main.src.api import WhiteAPI
from main.src.bridge import WhiteBridge
from main.src.db import Base, make_engine, make_session_factory
from main.src.pgstores import PgCandleStore
from main.src.service import (
    make_local_bridge,
    make_pg_white_api,
    make_white_bridge,
)


@pytest.fixture()
def sessions(tmp_path):
    """Session factory over a fresh SQLite database with all tables."""
    engine = make_engine(f"sqlite:///{tmp_path}/service.db")
    Base.metadata.create_all(engine)
    yield make_session_factory(engine)
    engine.dispose()


def _batch(n: int = 3) -> OhlcvBatch:
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


def _cmd(rid: str = "job-1") -> BacktestCommand:
    return BacktestCommand(
        request_id=rid, strategy_id="s1", dsl_entry="rsi.value < 30"
    )


class TestWhiteApiOverPg:
    """make_pg_white_api wires all three stores to one session factory."""

    def test_job_lifecycle_via_pg(self, sessions) -> None:
        api = make_pg_white_api(sessions)
        api.jobs.create(_cmd())

        api.handle_report(
            b'{"request_id":"job-1","report_json":"{\\"sharpe\\":1.1}",'
            b'"status":"completed","schema_version":1}'
        )
        status = api.get_backtest_status("job-1")
        assert status is not None
        assert status["status"] == "completed"
        assert status["report"]["sharpe"] == 1.1


class TestLocalServiceGlue:
    """md.ohlcv -> PG and cmd.backtest -> report, over a real broker."""

    @pytest.mark.asyncio
    async def test_ohlcv_persisted_idempotently(self, sessions) -> None:
        from faststream.rabbit import RabbitBroker, TestRabbitBroker
        from msgspec import json as msgspec_json

        broker = RabbitBroker()
        local = make_local_bridge(sessions, broker=broker)
        store = PgCandleStore(sessions)
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
        assert store.count("BTC-USDT") == 3  # duplicates collapsed

    @pytest.mark.asyncio
    async def test_backtest_report_lands_in_pg(self, sessions) -> None:
        """cmd -> stub runner -> evt.report -> PG job store (completed)."""
        from faststream.rabbit import RabbitBroker, TestRabbitBroker

        broker = RabbitBroker()
        api = make_pg_white_api(sessions)
        WhiteBridge(api, broker=broker)
        local = make_local_bridge(sessions, broker=broker)
        cmd = _cmd()
        async with TestRabbitBroker(broker):
            await local.broker.publish(
                msgspec_json.encode(cmd),
                queue=QueueName.CMD_BACKTEST.value,
            )
            assert api.jobs.get("job-1") is None  # phase split
            api.jobs.create(cmd)  # white accepted the job meanwhile
            await local.broker.publish(
                msgspec_json.encode(cmd),
                queue=QueueName.CMD_BACKTEST.value,
            )
        status = api.get_backtest_status("job-1")
        assert status is not None
        assert status["status"] == "completed"
        assert status["report"]["error"] == (
            "no local backtest runner configured"
        )

    @pytest.mark.asyncio
    async def test_missing_job_stays_tolerant(self, sessions) -> None:
        """Report for an unknown job must not crash the subscriber."""
        from faststream.rabbit import RabbitBroker, TestRabbitBroker
        from msgspec import json as msgspec_json

        broker = RabbitBroker()
        api = make_pg_white_api(sessions)
        WhiteBridge(api, broker=broker)
        local = make_local_bridge(sessions, broker=broker)
        async with TestRabbitBroker(broker):
            await local.broker.publish(
                msgspec_json.encode(_cmd("ghost")),
                queue=QueueName.CMD_BACKTEST.value,
            )
        assert api.jobs.get("ghost") is None


class TestWhiteBridgeWiring:
    """make_white_bridge equals direct construction (ACL checks run)."""

    def test_builds_with_defaults(self) -> None:
        bridge = make_white_bridge(WhiteAPI())
        assert bridge.api is not None
        assert bridge.broker is not None
