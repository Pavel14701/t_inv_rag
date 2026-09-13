"""Shared fixtures for the strategies test suite."""

from __future__ import annotations

import pytest

from strategies.src.application.strategy import Strategy
from strategies.src.application.types import PriceDataFramePolars


SAMPLE_MANIFEST = {
    "indicators": {
        "rsi": {"attributes": ["value"]},
        "close": {"attributes": []},
        "open": {"attributes": []},
        "high": {"attributes": []},
        "low": {"attributes": []},
        "ema": {
            "attributes": ["value"],
            "parameters": {"length": {"type": "float"}},
        },
        "rising": {},
    }
}


@pytest.fixture
def manifest() -> dict:
    """A minimal indicator manifest for validation."""
    return SAMPLE_MANIFEST


@pytest.fixture
def etalon() -> Strategy:
    """The etalon strategy from TZ-02 acceptance criteria."""
    return Strategy(
        id="etalon-1",
        name="RSI oversold + momentum",
        description="RSI.value < 30 with rising close",
        dsl_entry="rsi.value < 30",
        dsl_exit="close > open",
    )


@pytest.fixture
def ohlc_frame() -> PriceDataFramePolars:
    """A small OHLCV frame in the unified schema."""
    return PriceDataFramePolars(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100, 200, 300],
        }
    )
