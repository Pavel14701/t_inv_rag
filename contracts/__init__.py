"""Shared message contracts (TZ-09, TZ-08).

msgspec structs for all queue messages between the white API and the
local GPU node. These types are the single source of truth for the
communication protocol; no local copies allowed (TZ-09 item 1).
"""

from __future__ import annotations

from enum import Enum

import msgspec


# --------------------------------------------------------------------------- #
# Enums (from dev_docs/api.md)
# --------------------------------------------------------------------------- #


class Side(str, Enum):
    """Trade direction."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"


class TdMode(str, Enum):
    """Trading mode."""

    CROSS = "cross"
    ISOLATED = "isolated"


class PosSide(str, Enum):
    """Position side."""

    LONG = "long"
    SHORT = "short"
    NET = "net"


# --------------------------------------------------------------------------- #
# Data messages: white -> local
# --------------------------------------------------------------------------- #


class Candle(msgspec.Struct, frozen=True):
    """OHLCV candle (from market data ingest)."""

    inst_id: str
    ts: int  # unix timestamp ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    schema_version: int = 1


class OhlcvBatch(msgspec.Struct, frozen=True):
    """Batch of candles on the md.ohlcv queue."""

    inst_id: str
    candles: list[Candle]
    schema_version: int = 1


class AggBar(msgspec.Struct, frozen=True):
    """Aggregated bar on the md.agg queue."""

    inst_id: str
    anchor: str  # "D", "W", "H1", etc.
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Command messages: white -> local (ACL-restricted)
# --------------------------------------------------------------------------- #


class BacktestCommand(msgspec.Struct, frozen=True):
    """Request a backtest run on the local GPU node."""

    request_id: str
    strategy_id: str
    dsl_entry: str
    dsl_exit: str | None = None
    config_json: str = "{}"
    schema_version: int = 1


class TrainCommand(msgspec.Struct, frozen=True):
    """Request a model training run (ACL-protected)."""

    request_id: str
    config_json: str
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Event messages: local -> white
# --------------------------------------------------------------------------- #


class SignalEvent(msgspec.Struct, frozen=True):
    """Trading signal with optional P(win) on evt.signals queue."""

    inst_id: str
    ts: int
    direction: str  # "long" | "short"
    entry_price: float
    sl_price: float
    tp_price: float
    p_win: float | None = None  # P(win) crosses the boundary as a number
    strategy_id: str = ""
    schema_version: int = 1


class ReportEvent(msgspec.Struct, frozen=True):
    """Backtest/training report on evt.report queue."""

    request_id: str
    report_json: str
    status: str  # "completed" | "failed" | "running"
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Queue topology and ACL (TZ-09 items 2.3, 3)
# --------------------------------------------------------------------------- #


class QueueName(str, Enum):
    """Canonical queue names. Direction is enforced by ACL."""

    MD_OHLCV = "md.ohlcv"  # white -> local
    MD_AGG = "md.agg"  # white -> local
    CMD_BACKTEST = "cmd.backtest"  # white -> local
    CMD_TRAIN = "cmd.train"  # white -> local
    EVT_REPORT = "evt.report"  # local -> white
    EVT_SIGNALS = "evt.signals"  # local -> white


# Which queues each node may PUBLISH to
PUBLISH_ACL: dict[str, set[QueueName]] = {
    "white": {
        QueueName.MD_OHLCV,
        QueueName.MD_AGG,
        QueueName.CMD_BACKTEST,
        QueueName.CMD_TRAIN,
    },
    "local": {QueueName.EVT_REPORT, QueueName.EVT_SIGNALS},
}

# Which queues each node may CONSUME from
CONSUME_ACL: dict[str, set[QueueName]] = {
    "white": {QueueName.EVT_REPORT, QueueName.EVT_SIGNALS},
    "local": {
        QueueName.MD_OHLCV,
        QueueName.MD_AGG,
        QueueName.CMD_BACKTEST,
        QueueName.CMD_TRAIN,
    },
}


def can_publish(node: str, queue: QueueName) -> bool:
    """Check if a node is allowed to publish to a queue (ACL)."""
    return queue in PUBLISH_ACL.get(node, set())


def can_consume(node: str, queue: QueueName) -> bool:
    """Check if a node is allowed to consume from a queue (ACL)."""
    return queue in CONSUME_ACL.get(node, set())
