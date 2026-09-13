"""PostgreSQL-backed stores with the same interfaces as ``main.src.api``.

Each store takes a ``sessionmaker`` and implements the exact method set of
its in-memory counterpart, so ``WhiteAPI`` accepts either implementation.
"""

from __future__ import annotations

from typing import Any

from msgspec import json as msgspec_json
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from contracts import BacktestCommand, Candle, ReportEvent, SignalEvent

from .api import BacktestJob
from .db import BacktestJobRow, CandleRow, SignalRow


class PgCandleStore:
    """Idempotent candle store keyed by (inst_id, ts) - PostgreSQL."""

    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions

    def ingest(self, candle: Candle) -> bool:
        """Insert candle; True if new, False if duplicate."""
        with self._sessions() as s:
            key = {"inst_id": candle.inst_id, "ts": candle.ts}
            if s.get(CandleRow, key) is not None:
                return False
            s.add(
                CandleRow(
                    inst_id=candle.inst_id,
                    ts=candle.ts,
                    open=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                    volume=candle.volume,
                    schema_version=candle.schema_version,
                )
            )
            s.commit()
            return True

    def ingest_batch(self, batch: Any) -> int:
        """Ingest batch; returns count of NEW candles."""
        return sum(1 for c in batch.candles if self.ingest(c))

    def count(self, inst_id: str | None = None) -> int:
        """Count candles, optionally by instrument."""
        stmt = select(CandleRow)
        if inst_id:
            stmt = stmt.where(CandleRow.inst_id == inst_id)
        with self._sessions() as s:
            return len(s.scalars(stmt).all())


class PgJobStore:
    """Backtest job store - PostgreSQL."""

    def __init__(self, sessions: sessionmaker) -> None:
        self._sessions = sessions

    def create(self, cmd: BacktestCommand) -> BacktestJob:
        """Create a job from a command."""
        row = BacktestJobRow(
            job_id=cmd.request_id,
            status="pending",
            request_json=msgspec_json.encode(cmd).decode(),
        )
        with self._sessions() as s:
            s.merge(row)
            s.commit()
        return BacktestJob(
            job_id=cmd.request_id, status="pending", request=cmd
        )

    def get(self, job_id: str) -> BacktestJob | None:
        """Get a job by id; None if missing."""
        with self._sessions() as s:
            row = s.get(BacktestJobRow, job_id)
            if row is None:
                return None
            return self._to_job(row)

    def set_report(self, job_id: str, report: ReportEvent) -> None:
        """Attach report and mark job completed."""
        with self._sessions() as s:
            row = s.get(BacktestJobRow, job_id)
            if row is not None:
                row.report_json = msgspec_json.encode(report).decode()
                row.status = "completed"
                s.commit()

    @staticmethod
    def _to_job(row: BacktestJobRow) -> BacktestJob:
        """Rebuild the BacktestJob dataclass from a row."""
        request = (
            msgspec_json.decode(row.request_json, type=BacktestCommand)
            if row.request_json
            else None
        )
        report = (
            msgspec_json.decode(row.report_json, type=ReportEvent)
            if row.report_json
            else None
        )
        return BacktestJob(
            job_id=row.job_id,
            status=row.status,
            request=request,
            report=report,
        )


class PgSignalStore:
    """Signal store (latest N per instrument) - PostgreSQL."""

    def __init__(
        self, sessions: sessionmaker, max_per_inst: int = 100
    ) -> None:
        self._sessions = sessions
        self._max = max_per_inst

    def add(self, signal: SignalEvent) -> None:
        """Add a signal, evicting oldest rows over the per-instrument cap."""
        with self._sessions() as s:
            s.add(
                SignalRow(
                    inst_id=signal.inst_id,
                    ts=signal.ts,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    sl_price=signal.sl_price,
                    tp_price=signal.tp_price,
                    p_win=signal.p_win,
                    strategy_id=signal.strategy_id,
                    schema_version=signal.schema_version,
                )
            )
            s.commit()
            self._evict(s, signal.inst_id)

    def _evict(self, s: Session, inst_id: str) -> None:
        """Delete oldest rows beyond the cap for one instrument."""
        stmt = (
            select(SignalRow.id)
            .where(SignalRow.inst_id == inst_id)
            .order_by(SignalRow.id.desc())
        )
        ids = list(s.scalars(stmt).all())
        for old_id in ids[self._max :]:
            row = s.get(SignalRow, old_id)
            if row is not None:
                s.delete(row)
        s.commit()

    def latest(self, inst_id: str, limit: int = 10) -> list[SignalEvent]:
        """Return latest signals for an instrument (newest first)."""
        stmt = (
            select(SignalRow)
            .where(SignalRow.inst_id == inst_id)
            .order_by(SignalRow.id.desc())
            .limit(limit)
        )
        with self._sessions() as s:
            rows = s.scalars(stmt).all()
            return [
                SignalEvent(
                    inst_id=r.inst_id,
                    ts=r.ts,
                    direction=r.direction,
                    entry_price=r.entry_price,
                    sl_price=r.sl_price,
                    tp_price=r.tp_price,
                    p_win=r.p_win,
                    strategy_id=r.strategy_id,
                    schema_version=r.schema_version,
                )
                for r in rows
            ]
