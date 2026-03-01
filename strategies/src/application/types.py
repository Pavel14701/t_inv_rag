from typing import Any, Sequence

import polars as pl


class PriceDataFramePolars:
    """
    Polars-версия кастомного DataFrame с обязательными колонками
    и доступом к ним через атрибуты.
    """

    REQUIRED_COLUMNS: Sequence[str] = (
        "date",
        "open_price",
        "close_price",
        "high_price",
        "low_price",
        "volume",
        "turnover"
    )

    def __init__(
        self,
        data: pl.DataFrame | dict[str, Any],
        columns: Sequence[str] | None = None
    ) -> None:
        if isinstance(data, dict):
            self._data = pl.DataFrame(data, schema=columns)
        else:
            self._data = data

        self._validate_columns()
        self._convert_types()
        self._data = self._data.sort("date")

    def _validate_columns(self) -> None:
        missing = [
            col for col in self.REQUIRED_COLUMNS if col not in self._data.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _convert_types(self) -> None:
        self._data = self._data.with_columns([
            pl.col("date").str.to_datetime(),
            pl.col("open_price").cast(pl.Float64),
            pl.col("close_price").cast(pl.Float64),
            pl.col("high_price").cast(pl.Float64),
            pl.col("low_price").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
            pl.col("turnover").cast(pl.Float64),
        ])

    @property
    def date(self) -> pl.Series:
        return self._data["date"]

    @property
    def open_price(self) -> pl.Series:
        return self._data["open_price"]

    @property
    def close_price(self) -> pl.Series:
        return self._data["close_price"]

    @property
    def high_price(self) -> pl.Series:
        return self._data["high_price"]

    @property
    def low_price(self) -> pl.Series:
        return self._data["low_price"]

    @property
    def volume(self) -> pl.Series:
        return self._data["volume"]

    @property
    def turnover(self) -> pl.Series:
        return self._data["turnover"]

    @property
    def df(self) -> pl.DataFrame:
        return self._data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)

    def __repr__(self) -> str:
        return repr(self._data)