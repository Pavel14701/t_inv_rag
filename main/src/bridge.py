"""FastStream RabbitMQ bridge (TZ-09 волна 2).

White side: publishes md.*/cmd.*, consumes evt.*.
Local side: consumes md.*/cmd.*, publishes evt.*.
Reconnect/backoff, heartbeat monitoring, schema-version tolerance.
"""

from __future__ import annotations

import asyncio
import json
import time

from collections.abc import Awaitable, Callable
from typing import Any

import msgspec

from faststream.rabbit import RabbitBroker, RabbitQueue
from msgspec import json as msgspec_json

from contracts import (
    BacktestCommand,
    OhlcvBatch,
    QueueName,
    ReportEvent,
    SignalEvent,
    can_consume,
    can_publish,
)


CURRENT_SCHEMA_VERSION = 1


class ReconnectPolicy:
    """Exponential backoff for broker reconnects."""

    def __init__(
        self, base_delay: float = 0.5, max_delay: float = 30.0,
        factor: float = 2.0, max_attempts: int = 10,
    ) -> None:
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.factor = factor
        self.max_attempts = max_attempts

    def delay_for(self, attempt: int) -> float:
        """Delay before retry number ``attempt`` (0-based)."""
        return min(self.base_delay * (self.factor**attempt), self.max_delay)


class HeartbeatMonitor:
    """Tracks last-seen timestamps per queue for lag monitoring."""

    def __init__(self) -> None:
        self._last_seen: dict[str, float] = {}

    def record(self, queue: str, ts: float | None = None) -> None:
        """Record a message seen on ``queue`` (monotonic ts)."""
        self._last_seen[queue] = time.monotonic() if ts is None else ts

    def last_seen(self, queue: str) -> float | None:
        """Return last-seen monotonic ts for ``queue``."""
        return self._last_seen.get(queue)

    def stale_queues(self, max_age_s: float) -> list[str]:
        """Queues silent longer than ``max_age_s``."""
        now = time.monotonic()
        return [
            q for q, ts in self._last_seen.items() if now - ts > max_age_s
        ]


def _check_version(body: dict[str, Any]) -> bool:
    """True if message schema version is supported."""
    return body.get("schema_version", 1) <= CURRENT_SCHEMA_VERSION


class WhiteBridge:
    """White-node bridge: publish md.*/cmd.*, consume evt.*."""

    def __init__(
        self, api: Any, broker: RabbitBroker | None = None,
        monitor: HeartbeatMonitor | None = None,
    ) -> None:
        self.api = api
        self.broker = broker or RabbitBroker()
        self.monitor = monitor or HeartbeatMonitor()
        self._setup_subscribers()

    def _setup_subscribers(self) -> None:
        for q in (QueueName.EVT_REPORT, QueueName.EVT_SIGNALS):
            if not can_consume("white", q):
                raise PermissionError(f"white cannot consume {q}")

        @self.broker.subscriber(
            RabbitQueue(QueueName.EVT_REPORT.value, durable=True)
        )
        def on_report(data: dict) -> None:
            self.monitor.record(QueueName.EVT_REPORT.value)
            self.api.handle_report(msgspec_json.encode(data))

        @self.broker.subscriber(
            RabbitQueue(QueueName.EVT_SIGNALS.value, durable=True)
        )
        def on_signal(data: dict) -> None:
            self.monitor.record(QueueName.EVT_SIGNALS.value)
            self.api.handle_signal(msgspec_json.encode(data))

    def _publish_queue_check(self, queue: QueueName) -> None:
        if not can_publish("white", queue):
            raise PermissionError(f"white cannot publish to {queue}")

    async def publish_ohlcv(self, batch: OhlcvBatch) -> None:
        """Publish a candle batch to md.ohlcv."""
        self._publish_queue_check(QueueName.MD_OHLCV)
        await self.broker.publish(
            msgspec_json.encode(batch), queue=QueueName.MD_OHLCV.value
        )

    async def publish_backtest(self, cmd: BacktestCommand) -> None:
        """Publish a backtest command to cmd.backtest (202 flow)."""
        self._publish_queue_check(QueueName.CMD_BACKTEST)
        await self.broker.publish(
            msgspec_json.encode(cmd), queue=QueueName.CMD_BACKTEST.value
        )

    async def connect(self, url: str, policy: ReconnectPolicy) -> None:
        """Connect with exponential-backoff retries."""
        last_exc: Exception | None = None
        for attempt in range(max(policy.max_attempts, 1)):
            try:
                await self.broker.connect(url)  # type: ignore[call-arg]
                return
            except Exception as exc:
                last_exc = exc
                await asyncio.sleep(policy.delay_for(attempt))
        if last_exc is not None:
            raise last_exc


class LocalBridge:
    """Local GPU-node bridge: consume md.*/cmd.*, publish evt.*."""

    def __init__(
        self,
        broker: RabbitBroker | None = None,
        monitor: HeartbeatMonitor | None = None,
        on_candles: Callable[[OhlcvBatch], Awaitable[None]] | None = None,
        on_backtest: (
            Callable[[BacktestCommand], Awaitable[ReportEvent]] | None
        ) = None,
    ) -> None:
        self.broker = broker or RabbitBroker()
        self.monitor = monitor or HeartbeatMonitor()
        self._on_candles = on_candles
        self._on_backtest = on_backtest
        self._setup_subscribers()

    def _setup_subscribers(self) -> None:
        for q in (QueueName.EVT_REPORT, QueueName.EVT_SIGNALS):
            if not can_publish("local", q):
                raise PermissionError(f"local cannot publish to {q}")

        @self.broker.subscriber(
            RabbitQueue(QueueName.MD_OHLCV.value, durable=True)
        )
        async def on_ohlcv(data: dict) -> None:
            self.monitor.record(QueueName.MD_OHLCV.value)
            if not _check_version(data):
                return
            batch = msgspec.convert(data, type=OhlcvBatch)
            if self._on_candles is not None:
                await self._on_candles(batch)

        @self.broker.subscriber(
            RabbitQueue(QueueName.CMD_BACKTEST.value, durable=True)
        )
        async def on_backtest(data: dict) -> None:
            self.monitor.record(QueueName.CMD_BACKTEST.value)
            if not _check_version(data):
                return
            cmd = msgspec.convert(data, type=BacktestCommand)
            if self._on_backtest is None:
                return
            report = await self._on_backtest(cmd)
            await self.publish_report(report)

    def _publish_queue_check(self, queue: QueueName) -> None:
        if not can_publish("local", queue):
            raise PermissionError(f"local cannot publish to {queue}")

    async def publish_report(self, report: ReportEvent) -> None:
        """Publish a backtest/training report to evt.report."""
        self._publish_queue_check(QueueName.EVT_REPORT)
        await self.broker.publish(
            msgspec_json.encode(report), queue=QueueName.EVT_REPORT.value
        )

    async def publish_signal(self, signal: SignalEvent) -> None:
        """Publish a trading signal to evt.signals."""
        self._publish_queue_check(QueueName.EVT_SIGNALS)
        await self.broker.publish(
            msgspec_json.encode(signal), queue=QueueName.EVT_SIGNALS.value
        )


def report_from_payload(
    request_id: str, status: str, payload: dict[str, Any]
) -> ReportEvent:
    """Build a ReportEvent from a payload dict."""
    return ReportEvent(
        request_id=request_id,
        report_json=json.dumps(payload),
        status=status,
    )
