"""Unified OHLC data frame for the strategy layer (TZ-02 item 2.1).

Canonical column names across the whole system:
``date, open, high, low, close, volume``.  ``turnover`` is accepted
when present but is not required (legacy parquet side-car).
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl


class PriceDataFramePolars:
    """Polars frame with required OHLCV columns and attribute access.

    Column names follow the unified schema (TZ-02 item 2.1):
    ``open, high, low, close`` (not the legacy ``*__price`` aliases).
    ``infer.data.normalize`` maps the legacy names onto these on load.
    """

    REQUIRED_COLUMNS: Sequence[str] = (
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    def __init__(
        self,
        data: pl.DataFrame | dict[str, Any],
        columns: Sequence[str] | None = None,
    ) -> None:
        if isinstance(data, dict):
            self._data = pl.DataFrame(data, schema=columns)
        else:
            self._data = data

        self._normalize_legacy_columns()
        self._validate_columns()
        self._convert_types()
        self._data = self._data.sort("date")

    def _normalize_legacy_columns(self) -> None:
        """Map legacy ``*__price`` column names onto the unified schema."""
        rename = {
            "open_price": "open",
            "close_price": "close",
            "high_price": "high",
            "low_price": "low",
        }
        to_apply = {
            old: new
            for old, new in rename.items()
            if old in self._data.columns
        }
        if to_apply:
            self._data = self._data.rename(to_apply)

    def _validate_columns(self) -> None:
        missing = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in self._data.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _convert_types(self) -> None:
        self._data = self._data.with_columns(
            [
                pl.col("date").str.to_datetime(),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Int64),
            ]
        )

    @property
    def date(self) -> pl.Series:
        """Date column as a Series."""
        return self._data["date"]

    @property
    def open(self) -> pl.Series:
        """Open price column as a Series."""
        return self._data["open"]

    @property
    def close(self) -> pl.Series:
        """Close price column as a Series."""
        return self._data["close"]

    @property
    def high(self) -> pl.Series:
        """High price column as a Series."""
        return self._data["high"]

    @property
    def low(self) -> pl.Series:
        """Low price column as a Series."""
        return self._data["low"]

    @property
    def volume(self) -> pl.Series:
        """Volume column as a Series."""
        return self._data["volume"]

    @property
    def turnover(self) -> pl.Series | None:
        """Turnover column as a Series, if present (optional side-car)."""
        return (
            self._data["turnover"]
            if "turnover" in self._data.columns
            else None
        )

    @property
    def df(self) -> pl.DataFrame:
        """The underlying Polars DataFrame."""
        return self._data

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the inner DataFrame."""
        return getattr(self._data, name)

    def __repr__(self) -> str:
        """Return the repr of the inner DataFrame."""
        return repr(self._data)
