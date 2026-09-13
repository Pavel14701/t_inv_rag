"""PG-backed stores, models and Alembic migrations (TZ-10 wave 2).

SQLite is used as the test backend: the schema is deliberately portable
(TEXT for JSON payloads, no PG-specific column types).
"""

from __future__ import annotations

import pytest

from sqlalchemy import inspect

from contracts import (
    BacktestCommand,
    Candle,
    ReportEvent,
    SignalEvent,
)
from main.src.api import WhiteAPI
from main.src.db import (
    Base,
    make_engine,
    make_session_factory,
)
from main.src.pgstores import (
    PgCandleStore,
    PgJobStore,
    PgSignalStore,
)


@pytest.fixture()
def sessions(tmp_path):
    """Session factory over a fresh SQLite database with all tables."""
    engine = make_engine(f"sqlite:///{tmp_path}/white.db")
    Base.metadata.create_all(engine)
    yield make_session_factory(engine)
    engine.dispose()


def _candle(ts: int = 1000) -> Candle:
    return Candle(
        inst_id="SBER",
        ts=ts,
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10.0,
    )


def _cmd(rid: str = "req-1") -> BacktestCommand:
    return BacktestCommand(
        request_id=rid, strategy_id="s1", dsl_entry="close > open"
    )


def _report(rid: str = "req-1") -> ReportEvent:
    return ReportEvent(
        request_id=rid, report_json='{"sharpe": 1.2}', status="completed"
    )


def _signal(ts: int = 1000, inst: str = "SBER") -> SignalEvent:
    return SignalEvent(
        inst_id=inst,
        ts=ts,
        direction="long",
        entry_price=1.5,
        sl_price=1.4,
        tp_price=1.8,
        p_win=0.6,
    )


class TestPgCandleStore:
    """PgCandleStore mirrors the in-memory CandleStore contract."""

    def test_ingest_idempotent(self, sessions) -> None:
        store = PgCandleStore(sessions)
        assert store.ingest(_candle()) is True
        assert store.ingest(_candle()) is False
        assert store.count() == 1
        assert store.count("SBER") == 1
        assert store.count("LKOH") == 0

    def test_ingest_batch_counts_new(self, sessions) -> None:
        from contracts import OhlcvBatch

        store = PgCandleStore(sessions)
        batch = OhlcvBatch(
            inst_id="SBER", candles=[_candle(1), _candle(2), _candle(3)]
        )
        assert store.ingest_batch(batch) == 3
        assert store.ingest_batch(batch) == 0


class TestPgJobStore:
    """PgJobStore mirrors the in-memory JobStore contract."""

    def test_lifecycle(self, sessions) -> None:
        store = PgJobStore(sessions)
        job = store.create(_cmd())
        assert job.status == "pending"
        assert store.get("req-1") is not None
        assert store.get("nope") is None

        store.set_report("req-1", _report())
        done = store.get("req-1")
        assert done is not None
        assert done.status == "completed"
        assert done.report is not None
        assert done.report.report_json == '{"sharpe": 1.2}'
        assert done.request is not None
        assert done.request.dsl_entry == "close > open"

    def test_set_report_missing_is_noop(self, sessions) -> None:
        store = PgJobStore(sessions)
        store.set_report("ghost", _report("ghost"))


class TestPgSignalStore:
    """PgSignalStore mirrors the in-memory SignalStore contract."""

    def test_latest_newest_first(self, sessions) -> None:
        store = PgSignalStore(sessions)
        for ts in (1, 2, 3):
            store.add(_signal(ts=ts))
        latest = store.latest("SBER", limit=2)
        assert [s.ts for s in latest] == [3, 2]

    def test_eviction_cap(self, sessions) -> None:
        store = PgSignalStore(sessions, max_per_inst=3)
        for ts in range(5):
            store.add(_signal(ts=ts))
        assert len(store.latest("SBER", limit=100)) == 3
        assert [s.ts for s in store.latest("SBER")] == [4, 3, 2]


class TestWhiteApiWithPgStores:
    """WhiteAPI accepts PG stores - same handlers as in-memory."""

    def test_report_roundtrip(self, sessions) -> None:
        from msgspec import json as msgspec_json

        api = WhiteAPI(
            jobs=PgJobStore(sessions),
            signals=PgSignalStore(sessions),
        )
        api.jobs.create(_cmd())
        api.handle_signal(msgspec_json.encode(_signal()))
        api.handle_report(msgspec_json.encode(_report()))
        status = api.get_backtest_status("req-1")
        assert status is not None
        assert status["status"] == "completed"
        assert status["report"]["sharpe"] == 1.2
        assert api.get_signals("SBER")[0]["p_win"] == 0.6


class TestAlembicMigrations:
    """Initial migration creates the schema; downgrade removes it."""

    def test_upgrade_head_and_downgrade(self, tmp_path, monkeypatch):
        from alembic import command
        from alembic.config import Config

        db_path = tmp_path / "mig.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
        cfg = Config("alembic.ini")
        command.upgrade(cfg, "head")

        engine = make_engine(f"sqlite:///{db_path}")
        tables = set(inspect(engine).get_table_names())
        assert {"candles", "strategies", "backtest_jobs", "signals"} <= tables
        engine.dispose()

        command.downgrade(cfg, "base")
        engine = make_engine(f"sqlite:///{db_path}")
        tables = set(inspect(engine).get_table_names())
        assert "candles" not in tables
        engine.dispose()


class TestDatabaseDi:
    """DatabaseProvider wiring in the api contour."""

    def test_none_without_database_url(self, monkeypatch) -> None:
        from main.src.di import DatabasePort, build_container

        monkeypatch.delenv("DATABASE_URL", raising=False)
        container = build_container("api")
        port = container.get(DatabasePort)
        assert isinstance(port, DatabasePort)
        assert port.sessions is None

    def test_sqlite_url_builds_session_factory(
        self, tmp_path, monkeypatch
    ) -> None:
        from main.src.di import DatabasePort, build_container

        url = f"sqlite:///{tmp_path}/di.db"
        Base.metadata.create_all(make_engine(url))
        monkeypatch.setenv("DATABASE_URL", url)
        container = build_container("api")
        port = container.get(DatabasePort)
        assert port.sessions is not None
        store = PgCandleStore(port.sessions)
        assert store.ingest(_candle()) is True
