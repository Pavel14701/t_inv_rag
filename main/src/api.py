"""White API: queue consumers + REST data layer (TZ-10)."""

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Any

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


class CandleStore:
    """Idempotent candle store keyed by (inst_id, ts)."""

    def __init__(self) -> None:
        self._candles: dict[tuple[str, int], Candle] = {}

    def ingest(self, candle: Candle) -> bool:
        """Insert candle; True if new, False if duplicate."""
        key = (candle.inst_id, candle.ts)
        if key in self._candles:
            return False
        self._candles[key] = candle
        return True

    def ingest_batch(self, batch: OhlcvBatch) -> int:
        """Ingest batch; returns count of NEW candles."""
        return sum(1 for c in batch.candles if self.ingest(c))

    def count(self, inst_id: str | None = None) -> int:
        """Count candles, optionally by instrument."""
        if inst_id:
            return sum(1 for (iid, _) in self._candles if iid == inst_id)
        return len(self._candles)


@dataclass(slots=True)
class BacktestJob:
    """Async backtest job (202 + job_id pattern)."""

    job_id: str
    status: str = "pending"
    request: BacktestCommand | None = None
    report: ReportEvent | None = None


class JobStore:
    """In-memory job store (PostgreSQL in production)."""

    def __init__(self) -> None:
        self._jobs: dict[str, BacktestJob] = {}

    def create(self, cmd: BacktestCommand) -> BacktestJob:
        """Create a job from a command."""
        job = BacktestJob(job_id=cmd.request_id, request=cmd)
        self._jobs[cmd.request_id] = job
        return job

    def get(self, job_id: str) -> BacktestJob | None:
        """Get a job by id; None if missing."""
        return self._jobs.get(job_id)

    def set_report(self, job_id: str, report: ReportEvent) -> None:
        """Attach report and mark job completed."""
        if job_id in self._jobs:
            self._jobs[job_id].report = report
            self._jobs[job_id].status = "completed"


class SignalStore:
    """In-memory signal store (latest N per instrument)."""

    def __init__(self, max_per_inst: int = 100) -> None:
        self._signals: dict[str, list[SignalEvent]] = {}
        self._max = max_per_inst

    def add(self, signal: SignalEvent) -> None:
        """Add a signal, evicting oldest if over limit."""
        lst = self._signals.setdefault(signal.inst_id, [])
        lst.append(signal)
        if len(lst) > self._max:
            lst.pop(0)

    def latest(self, inst_id: str, limit: int = 10) -> list[SignalEvent]:
        """Return latest signals for an instrument (newest first)."""
        return list(reversed(self._signals.get(inst_id, [])[-limit:]))


class WhiteAPI:
    """White API service: stores + queue handlers + REST data."""

    def __init__(
        self,
        candles: Any | None = None,
        jobs: Any | None = None,
        signals: Any | None = None,
    ) -> None:
        self.candles = candles if candles is not None else CandleStore()
        self.jobs = jobs if jobs is not None else JobStore()
        self.signals = signals if signals is not None else SignalStore()

    def handle_report(self, raw: bytes) -> None:
        """Consumer for evt.report."""
        report = msgspec_json.decode(raw, type=ReportEvent)
        self.jobs.set_report(report.request_id, report)

    def handle_signal(self, raw: bytes) -> None:
        """Consumer for evt.signals."""
        signal = msgspec_json.decode(raw, type=SignalEvent)
        self.signals.add(signal)

    def submit_backtest(self, cmd: BacktestCommand) -> BacktestJob:
        """Submit async backtest (202 + job_id)."""
        if not can_publish("white", QueueName.CMD_BACKTEST):
            raise PermissionError("ACL violation")
        return self.jobs.create(cmd)

    def get_backtest_status(self, job_id: str) -> dict[str, Any] | None:
        """GET /backtests/{job_id} data."""
        job = self.jobs.get(job_id)
        if job is None:
            return None
        result: dict[str, Any] = {"job_id": job.job_id, "status": job.status}
        if job.report:
            result["report"] = json.loads(job.report.report_json)
        return result

    def get_signals(self, inst_id: str, limit: int = 10) -> list[dict]:
        """GET /signals data."""
        return [
            {
                "inst_id": s.inst_id,
                "ts": s.ts,
                "direction": s.direction,
                "p_win": s.p_win,
            }
            for s in self.signals.latest(inst_id, limit)
        ]
