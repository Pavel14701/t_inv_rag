import talib as ta # type: ignore
from pandas import DataFrame # type: ignore
from talib._ta_lib import MA_Type # type: ignore

from domain.entities import StochRsiConfigDM
from infrastructure._types import PriceDataFrame


class StochRSI:
    """
    Stochastic RSI (StochRSI) Indicator Implementation.

    The Stochastic RSI measures the relative position of RSI
    compared to its high-low range over a defined period. 
    It is commonly used to identify potential overbought 
    and oversold conditions.

    Formula:
    1. Calculate the **Relative Strength Index (RSI)** over `timeperiod`.
    2. Compute the **lowest and highest RSI values** over the `fastk_period` window.
    3. Normalize RSI within this range:
       - %K = (RSI - Lowest RSI) / (Highest RSI - Lowest RSI) * 100.
    4. Apply a smoothing Moving Average (`fastd_period`) to %K to get **%D**.
    5. Generate trading signals based on crossings:
       - %K crosses above %D in oversold conditions (**Buy Signal**).
       - %K crosses below %D in overbought conditions (**Sell Signal**).

    Usage:
    - Provides **momentum-based trading signals**.
    - Helps detect **trend reversals in extreme conditions (oversold/overbought)**.
    """

    def calculate_stochrsi(
            self, 
            data: PriceDataFrame, 
            config: StochRsiConfigDM
        ) -> DataFrame:
        """
        Computes the Stochastic RSI (%K and %D) values.
        Args:
            data (PriceDataFrame): The price dataset containing closing prices.
            config (StochRsiConfigDM): Configuration for Stochastic RSI 
                (time period, smoothing factors).
        Returns:
            DataFrame: A DataFrame containing the %K and %D values.
        """
        # Compute the Stochastic RSI components using TA-Lib
        fastk, fastd = ta.STOCHRSI(  # type: ignore
            close=data.close_prices,  # type: ignore
            # RSI calculation period
            timeperiod=config.timeperiod,
            # %K period (high-low RSI range)
            fastk_period=config.fastk_period,
            # %D smoothing period  
            fastd_period=config.fastd_period, 
            # Moving Average type for %D 
            fastd_matype=MA_Type(config.fastd_matype)  
        )
        return DataFrame({"fastk": fastk, "fastd": fastd}, index=data.index)  # type: ignore

    def create_signals(self, data: DataFrame) -> DataFrame:
        """
        Generates buy/sell signals based on Stochastic RSI crossovers.
        Signals:
        - Buy: When %K crosses above %D in oversold 
            conditions (**fastk < 20**).
        - Sell: When %K crosses below %D in overbought 
            conditions (**fastk > 80**).
        Args:
            data (DataFrame): DataFrame containing %K and %D values.
        Returns:
            DataFrame: A DataFrame with separate buy and sell signal series.
        """
        # Buy signal: %K crosses above %D **and** is 
        # in oversold territory (<20)
        buy_signals = (
            data['fastk'] > data['fastd']
        ) & (
            data['fastk'].shift(1) < data['fastd'].shift(1)  # type: ignore
        ) & (
            data['fastk'] < 20
        )
        # Sell signal: %K crosses below %D **and** is 
        # in overbought territory (>80)
        sell_signals = (
            data['fastk'] < data['fastd']
        ) & (
            data['fastk'].shift(1) > data['fastd'].shift(1)  # type: ignore
        ) & (
            data['fastk'] > 80
        )
        return DataFrame(
            {
                "buy_signals": buy_signals.astype(int), 
                "sell_signals": sell_signals.astype(int)
            }, 
            index=data.index  # type: ignore
        )

    def get_last_signal(
        self, 
        data: PriceDataFrame, 
        config: StochRsiConfigDM
    ) -> str | None:
        """
        Determines the last trading signal based on Stochastic RSI.
        - If the last bar has a buy signal → returns "long".
        - If the last bar has a sell signal → returns "short".
        - If no signal is detected → returns None.
        Args:
            data (PriceDataFrame): Market data with closing prices.
            config (StochRsiConfigDM): Stochastic RSI configuration.
        Returns:
            str | None: "long" if buy signal, "short" if sell signal,
              or None if no signal is present.
        """
        # Compute Stochastic RSI values
        stoch_rsi_df = self.calculate_stochrsi(data, config)
        # Generate trading signals
        signals_df = self.create_signals(stoch_rsi_df)
        # Check the last signal and return appropriate action
        if signals_df["buy_signals"].iloc[-1] == 1:  # type: ignore
            return "long"
        elif signals_df["sell_signals"].iloc[-1] == 1:  # type: ignore
            return "short"
        return None
