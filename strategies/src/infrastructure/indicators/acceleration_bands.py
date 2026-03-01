import numpy as np
import pandas as pd # type: ignore
from numpy.typing import NDArray
from pandas_ta import accbands  # type: ignore

from domain.entities import AcceletrationBandsDM
from infrastructure._types import PriceDataFrame


class AccelerationBands:
    """
    Acceleration Bands Indicator Implementation.

    Acceleration Bands (ACCB) were developed by Mark Helweg and 
    are designed to indicate volatility and acceleration in 
    price movements. These bands consist of an upper band, 
    lower band, and centerline derived from a moving average.

    Usage:
    - Detect periods of increasing or decreasing price acceleration
    - Identify potential breakouts when price breaches the bands
    - Generate buy/sell signals based on band interaction
    """

    def calc_accbands(
        self,
        data: PriceDataFrame,
        config: AcceletrationBandsDM
    ) -> pd.DataFrame:
        """
        Calculates Acceleration Bands (ACCB) values.

        Args:
            data (PriceDataFrame): Price dataset containing high, low, and close prices.
            config (AcceletrationBandsDM): Configuration for Acceleration Bands 
            (length, smoothing method, drift, offset).

        Returns:
            pd.DataFrame: A DataFrame containing upper, lower, and center 
            acceleration bands, plus close prices.
        """
        acc_bands = accbands(
            high=data.high_prices,
            low=data.low_prices,
            close=data.close_prices,
            length=config.length,
            drift=config.drift,
            mamode=config.mamode,
            offset=config.offset,
        )
        if acc_bands is not None:
            acc_bands.index = pd.DatetimeIndex(data.index)  # type: ignore
            acc_bands["close_prices"] = data.close_prices
            return acc_bands
        raise ValueError("Failed to compute Acceleration Bands — result is None")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals based on Acceleration Bands interaction.

        Signals:
        - Buy: Price crosses above the lower band
        - Sell: Price crosses below the upper band

        Args:
            data (pd.DataFrame): DataFrame containing Acceleration Bands 
            and close prices.

        Returns:
            pd.DataFrame: A DataFrame containing buy_signals and sell_signals (0 or 1).
        """
        upper_band: NDArray[np.float64] = data["ACCbands_upper"].to_numpy(  # type: ignore
            dtype=np.float64
        )
        lower_band: NDArray[np.float64] = data["ACCbands_lower"].to_numpy(  # type: ignore
            dtype=np.float64
        )
        close_prices: NDArray[np.float64] = data["close_prices"].to_numpy(  # type: ignore
            dtype=np.float64
            )
        buy_signal: NDArray[np.int_] = (
            (close_prices < upper_band) & (np.roll(close_prices, 1) > upper_band)
        ).astype(np.int_)
        sell_signal: NDArray[np.int_] = (
            (close_prices > lower_band) & (np.roll(close_prices, 1) < lower_band)
        ).astype(np.int_)
        signals = pd.DataFrame(index=pd.DatetimeIndex(data.index))  # type: ignore
        signals["buy_signals"] = buy_signal
        signals["sell_signals"] = sell_signal
        return signals

    def check_last_signal(
        self,
        data: PriceDataFrame,
        config: AcceletrationBandsDM
    ) -> str | None:
        """
        Determines the last trading signal based on Acceleration Bands.

        Returns:
        - "long" if the last bar has a buy signal
        - "short" if the last bar has a sell signal
        - None if no signal is present

        Args:
            data (PriceDataFrame): Market data with price information.
            config (AcceletrationBandsDM): Configuration for Acceleration Bands.

        Returns:
            str | None: "long", "short", or None
        """
        acc_bands_df = self.calc_accbands(data, config)
        signals_df = self.generate_signals(acc_bands_df)
        buy_series: pd.Series[int] = signals_df["buy_signals"].astype("int64")
        sell_series: pd.Series[int] = signals_df["sell_signals"].astype("int64")
        if buy_series.iloc[-1] == 1:
            return "long"
        elif sell_series.iloc[-1] == 1:
            return "short"
        return None
