"""Tests for the unified OHLC schema (TZ-02 item 2.1)."""

from __future__ import annotations

import pytest

from strategies.src.application.types import PriceDataFramePolars


class TestUnifiedSchema:
    def test_short_column_names_accepted(
        self, ohlc_frame: PriceDataFramePolars
    ) -> None:
        assert list(ohlc_frame.REQUIRED_COLUMNS) == [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert ohlc_frame.open.to_list() == [10.0, 11.0, 12.0]
        assert ohlc_frame.close.to_list() == [10.5, 11.5, 12.5]

    def test_legacy_column_names_renamed(self) -> None:
        """Legacy ``*__price`` columns are mapped onto the unified schema."""
        frame = PriceDataFramePolars(
            {
                "date": ["2024-01-01"],
                "open_price": [10.0],
                "high_price": [11.0],
                "low_price": [9.0],
                "close_price": [10.5],
                "volume": [100],
            }
        )
        assert frame.open[0] == 10.0
        assert frame.close[0] == 10.5

    def test_missing_required_column_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing required columns"):
            PriceDataFramePolars({"date": ["2024-01-01"], "open": [10.0]})

    def test_turnover_optional(self, ohlc_frame: PriceDataFramePolars) -> None:
        assert ohlc_frame.turnover is None
