from typing import Literal

import numpy as np
import pandas as pd # type: ignore
import pandas_ta as ta # type: ignore
from numba import njit # type: ignore

from infrastructure._types import PriceDataFrame


class OTTIndicator:
    def __init__(
        self,
        length: int = 2,
        percent: float = 1.4,
        ma_type: Literal[
            "SMA", "EMA", "WMA", "TMA", "VAR", "WWMA", "ZLEMA", "TSF"
        ] = "VAR"
    ) -> None:
        self.length = length
        self.percent = percent
        self.ma_type = ma_type

    def calculate(self, df: PriceDataFrame) -> pd.DataFrame:
        price = df.close_prices
        ma = self._get_moving_average(price)
        long_stop, short_stop = self._compute_stop_levels(ma)
        direction = self._compute_trend_direction(
            ma.to_numpy(),
            long_stop.to_numpy(),
            short_stop.to_numpy()
        )
        base_stop = np.where(direction == 1, long_stop, short_stop)
        ott = np.where(
            ma > base_stop,
            base_stop * (200 + self.percent) / 200,
            base_stop * (200 - self.percent) / 200
        )
        result = pd.DataFrame({
            "price": price,
            "ma": ma,
            "long_stop": long_stop,
            "short_stop": short_stop,
            "direction": direction,
            "ott": ott
        }, index=df.index)
        return result

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df["price"]
        ott = df["ott"]
        ma = df["ma"]
        signals = pd.DataFrame(index=df.index)
        signals["buy_price_cross"] = (price > ott) & (price.shift(1) <= ott.shift(1))
        signals["sell_price_cross"] = (price < ott) & (price.shift(1) >= ott.shift(1))
        signals["buy_support_cross"] = (ma > ott) & (ma.shift(1) <= ott.shift(1))
        signals["sell_support_cross"] = (ma < ott) & (ma.shift(1) >= ott.shift(1))
        signals["buy_color_change"] = ott > ott.shift(1)
        signals["sell_color_change"] = ott < ott.shift(1)
        signals["signal"] = np.select(
            [
                signals["buy_price_cross"],
                signals["buy_support_cross"],
                signals["buy_color_change"],
                signals["sell_price_cross"],
                signals["sell_support_cross"],
                signals["sell_color_change"]
            ],
            [
                "buy_price_cross",
                "buy_support_cross",
                "buy_color_change",
                "sell_price_cross",
                "sell_support_cross",
                "sell_color_change"
            ],
            default=np.nan
        )
        return signals

    def get_last_signal(self, signals: pd.DataFrame) -> str | None:
        last = signals["signal"].dropna()
        return last.iloc[-1] if not last.empty else None

    def _get_moving_average(self, price: pd.Series) -> pd.Series:
        if self.ma_type == "SMA":
            return ta.sma(price, self.length) # type: ignore
        elif self.ma_type == "EMA":
            return ta.ema(price, self.length) # type: ignore
        elif self.ma_type == "WMA":
            return ta.wma(price, self.length) # type: ignore
        elif self.ma_type == "TMA":
            return ta.sma( # type: ignore
                ta.sma( # type: ignore
                    price, int(np.ceil(self.length / 2))), int(np.floor(self.length / 2)
                ) + 1)
        elif self.ma_type == "VAR":
            return self._calculate_vidya(price)
        elif self.ma_type == "WWMA":
            return self._calculate_wwma(price)
        elif self.ma_type == "ZLEMA":
            lag = int(self.length / 2)
            zlema_data = price + (price - price.shift(lag))
            return ta.ema(zlema_data, self.length) # type: ignore
        elif self.ma_type == "TSF":
            lrc = ta.linreg(price, self.length) # type: ignore
            lrc1 = lrc.shift(1)
            return lrc + (lrc - lrc1)
        else:
            raise ValueError(f"Unknown MA type: {self.ma_type}")

    def _calculate_vidya(self, price: pd.Series) -> pd.Series:
        alpha = 2 / (self.length + 1)
        up = np.where(price > price.shift(1), price - price.shift(1), 0)
        down = np.where(price < price.shift(1), price.shift(1) - price, 0)
        up_sum = pd.Series(up).rolling(9).sum()
        down_sum = pd.Series(down).rolling(9).sum()
        cmo = (up_sum - down_sum) / (up_sum + down_sum)
        vidya = np.zeros(len(price))
        for i in range(1, len(price)):
            vidya[i] = alpha * abs(cmo[i]) * price[i] + (
                1 - alpha * abs(cmo[i])
            ) * vidya[i - 1]
        return pd.Series(vidya, index=price.index)

    def _calculate_wwma(self, price: pd.Series) -> pd.Series:
        alpha = 1 / self.length
        wwma = np.zeros(len(price))
        for i in range(1, len(price)):
            wwma[i] = alpha * price[i] + (1 - alpha) * wwma[i - 1]
        return pd.Series(wwma, index=price.index)

    def _compute_stop_levels(self, ma: pd.Series) -> tuple[pd.Series, pd.Series]:
        offset = ma * self.percent * 0.01
        long_stop = np.maximum(ma - offset, (ma - offset).shift(1).fillna(ma - offset))
        short_stop = np.minimum(ma + offset, (ma + offset).shift(1).fillna(ma + offset))
        return pd.Series(
               long_stop, index=ma.index
            ), pd.Series(
               short_stop, index=ma.index
            )

    @staticmethod
    @njit
    def _compute_trend_direction(
        ma: np.ndarray, 
        long_stop: np.ndarray, 
        short_stop: np.ndarray
    ) -> np.ndarray:
        direction = np.ones(len(ma), dtype=np.int8)
        for i in range(1, len(ma)):
            if direction[i - 1] == -1 and ma[i] > short_stop[i - 1]:
                direction[i] = 1
            elif direction[i - 1] == 1 and ma[i] < long_stop[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
        return direction
