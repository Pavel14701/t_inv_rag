"""Service composition glue (TZ-10 wave 2, TZ-09 wave 2).

Wires the FastStream bridges to PostgreSQL persistence:
- white side: ``WhiteAPI`` over PG stores (jobs/signals/candles);
- local side: ``md.ohlcv`` -> PG (idempotent), ``cmd.backtest`` -> an
  injected runner that replies with an ``evt.report``.

Sync SQLAlchemy calls from async subscribers run in a worker thread
(``asyncio.to_thread``) so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio

from collections.abc import Awaitable, Callable
from typing import Any

from faststream.rabbit import RabbitBroker

from contracts import BacktestCommand, OhlcvBatch, ReportEvent

from .api import WhiteAPI
from .bridge import LocalBridge, WhiteBridge, report_from_payload
from .db import sessionmaker
from .pgstores import PgCandleStore, PgJobStore, PgSignalStore


def make_pg_white_api(sessions: sessionmaker) -> WhiteAPI:
    """Build a WhiteAPI backed by PostgreSQL stores."""
    return WhiteAPI(
        candles=PgCandleStore(sessions),
        jobs=PgJobStore(sessions),
        signals=PgSignalStore(sessions),
    )


def candle_saver(
    sessions: sessionmaker,
) -> Callable[[OhlcvBatch], Awaitable[None]]:
    """Async subscriber handler persisting md.ohlcv batches to PG."""
    store = PgCandleStore(sessions)

    async def save(batch: OhlcvBatch) -> None:
        await asyncio.to_thread(store.ingest_batch, batch)

    return save


def stub_backtest_runner() -> Callable[
    [BacktestCommand], Awaitable[ReportEvent]
]:
    """Fallback runner: answers with a failed report, never silently."""

    async def run(cmd: BacktestCommand) -> ReportEvent:  # noqa: RUF029
        """Awaitable interface match for LocalBridge.on_backtest."""
        return report_from_payload(
            cmd.request_id,
            "failed",
            {"error": "no local backtest runner configured"},
        )

    return run


def make_white_bridge(
    api: WhiteAPI,
    broker: RabbitBroker | None = None,
    monitor: Any | None = None,
) -> WhiteBridge:
    """White-node FastStream app: publish md.*/cmd.*, consume evt.*."""
    return WhiteBridge(api, broker=broker, monitor=monitor)


def make_local_bridge(
    sessions: sessionmaker,
    broker: RabbitBroker | None = None,
    runner: Callable[[BacktestCommand], Awaitable[ReportEvent]] | None = None,
    monitor: Any | None = None,
) -> LocalBridge:
    """Local-node FastStream app: md.ohlcv -> PG, cmd.backtest -> runner."""
    return LocalBridge(
        broker=broker,
        monitor=monitor,
        on_candles=candle_saver(sessions),
        on_backtest=runner or stub_backtest_runner(),
    )
