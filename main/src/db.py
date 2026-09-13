"""SQLAlchemy models and engine factory for white API persistence (TZ-10).

Tables mirror the in-memory stores in ``main.src.api``: candles, strategies,
backtest jobs, signals. All JSON payloads are stored as TEXT so the schema
runs unchanged on PostgreSQL and SQLite (unit tests).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    """Timezone-aware UTC now (stored naive-UTC for portability)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base for all white API tables."""


class CandleRow(Base):
    """OHLCV candle keyed by (inst_id, ts) - idempotent ingest."""

    __tablename__ = "candles"

    inst_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)


class StrategyRow(Base):
    """Registered strategy (TZ-02 format) persisted for REST serving."""

    __tablename__ = "strategies"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    manifest_hash: Mapped[str] = mapped_column(String(64), default="")
    dsl_entry: Mapped[str] = mapped_column(Text, default="")
    dsl_exit: Mapped[str | None] = mapped_column(Text, nullable=True)
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class BacktestJobRow(Base):
    """Async backtest job (202 + job_id pattern)."""

    __tablename__ = "backtest_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    request_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SignalRow(Base):
    """Trading signal on evt.signals (latest-N per instrument)."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    inst_id: Mapped[str] = mapped_column(String(64))
    ts: Mapped[int] = mapped_column(BigInteger)
    direction: Mapped[str] = mapped_column(String(8), default="long")
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    sl_price: Mapped[float] = mapped_column(Float, default=0.0)
    tp_price: Mapped[float] = mapped_column(Float, default=0.0)
    p_win: Mapped[float | None] = mapped_column(Float, nullable=True)
    strategy_id: Mapped[str] = mapped_column(String(64), default="")
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


Index("ix_signals_inst_id_id", SignalRow.inst_id, SignalRow.id)
Index("ix_strategies_manifest", StrategyRow.manifest_hash)


def make_engine(database_url: str) -> Engine:
    """Create a sync engine (psycopg for PostgreSQL, sqlite for tests)."""
    url = database_url
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url, future=True)


def make_session_factory(engine: Engine) -> sessionmaker:
    """Create a session factory bound to the engine."""
    return sessionmaker(bind=engine, future=True, expire_on_commit=False)
