from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd # type: ignore
import pandas_ta as ta  # type: ignore
from numba import njit  # type: ignore
from numpy.typing import NDArray

from domain.entities import AvslConfigDM
from infrastructure._types import PriceDataFrame


class BaseAVS(ABC):
    """
    Abstract base class for adaptive volume-based support/resistance 
    indicators (AVSL/AVSR).

    This class encapsulates the shared logic for computing dynamic price levels based on
    volume-weighted price behavior, momentum, and volatility. It defines the 
    core pipeline:
    - Compute volume-price metrics (VWMA, SMA, VPCI)
    - Derive a dynamic adjustment factor (`price_v`)
    - Apply a volatility-based deviation
    - Smooth the final level using a simple moving average (SMA)

    Subclasses must define:
    - Which price series to use (low or high)
    - How to apply the adjustment and deviation (directional logic)
    - The output column name
    """
    @abstractmethod
    def _get_price_series(self, data: PriceDataFrame) -> pd.Series[Any]:
        """
        Selects the base price series for level calculation.

        Returns:
            pd.Series: Typically either `low_prices` (for support) or 
            `high_prices` (for resistance).
        """
        pass

    @abstractmethod
    def _output_column_name(self) -> str:
        """
        Returns the name of the output column in the resulting DataFrame.

        Returns:
            str: e.g., "avsl" or "avsr"
        """
        pass

    @abstractmethod
    def _adjust_formula(
        self,
        price_series: pd.Series[Any],
        price_v: NDArray[np.float64],
        deviation: pd.Series[Any]
    ) -> pd.Series:
        """
        Applies directional logic to combine the base price, volume adjustment, 
        and deviation.

        This method defines how the final level is constructed:
        - For support: price - price_v + deviation
        - For resistance: price + price_v - deviation

        Args:
            price_series (pd.Series): Base price series (low or high).
            price_v (np.ndarray): Volume-adjusted price influence.
            deviation (pd.Series): Volatility-based offset.

        Returns:
            pd.Series: Adjusted price series to be smoothed by SMA.
        """
        pass

    def calculate(
        self, 
        data: PriceDataFrame, 
        config: AvslConfigDM
    ) -> pd.DataFrame:
        """
        Computes the adaptive support or resistance level.

        This method orchestrates the full indicator calculation:
        1. Computes volume-based moving averages and derived metrics (VPCI)
        2. Calculates a dynamic adjustment factor (`price_v`) using 
        volume-price interaction
        3. Computes a volatility-based deviation term
        4. Applies directional logic to combine base price, adjustment, and deviation
        5. Smooths the result using a simple moving average (SMA)

        Args:
            data (PriceDataFrame): Market data with close, low/high, and volume series.
            config (AvslConfigDM): Configuration with moving average 
            lengths and deviation factor.

        Returns:
            pd.DataFrame: A DataFrame with a single column (e.g., "avsl" or "avsr") 
            indexed by time.
        """
        vw_f, vw_s, vpc, vpr, vm, vpci = self._compute_base_series(data, config)
        price_v = self._price_fun(data, vpc, vpr, vpci)
        deviation = np.float64(config.stand_div) * vpci * vm
        price_series = self._get_price_series(data)
        adjusted = self._adjust_formula(price_series, price_v, deviation)
        result = ta.sma(close=adjusted, length=config.length_slow, talib=True) # type: ignore
        if result is None:
            raise ValueError("SMA calculation failed")
        return pd.DataFrame({self._output_column_name(): result}, index=data.index)

    def _compute_base_series(
        self, 
        data: PriceDataFrame, 
        config: AvslConfigDM
    ) -> tuple[
        pd.Series[float],
        pd.Series[float],
        pd.Series[float],
        pd.Series[float],
        pd.Series[float],
        pd.Series[float]
    ]:
        """
        Computes the foundational time series for volume-price dynamics.

        This includes:
        - VWMA (Volume-Weighted Moving Average)
        - SMA (Simple Moving Average)
        - VPC (Volume Price Confirmation): VWMA - SMA
        - VPR (Volume-Price Ratio): VWMA_fast / SMA_fast
        - VM (Volume Momentum): SMA(volume_fast) / SMA(volume_slow)
        - VPCI (Volume Price Confirmation Index): VPC * VPR * VM

        Returns:
            Tuple of six pd.Series: (vwma_fast, vwma_slow, vpc, vpr, vm, vpci)
        """
        vw_ma_fast = cast(pd.Series, ta.vwma( # type: ignore
            data.close_prices, 
            data.volumes, 
            config.length_fast
        ))
        vw_ma_slow = cast(pd.Series, ta.vwma( # type: ignore
            data.close_prices, 
            data.volumes, 
            config.length_slow
        ))
        sma_fast = cast(pd.Series, ta.sma( # type: ignore
            data.close_prices, 
            config.length_fast, 
            talib=True
        ))
        sma_slow = cast(pd.Series, ta.sma( # type: ignore
            data.close_prices, 
            config.length_slow, 
            talib=True
        ))
        vol_fast = cast(pd.Series, ta.sma( # type: ignore
            data.volumes, 
            config.length_fast, 
            talib=True
        ))
        vol_slow = cast(pd.Series, ta.sma( # type: ignore
            data.volumes, 
            config.length_slow, 
            talib=True
        ))
        vpc = (vw_ma_slow - sma_slow).astype("float64")
        vpr = (vw_ma_fast / sma_fast).astype("float64")
        vm = (vol_fast / vol_slow).astype("float64")
        vpci = (vpc * vpr * vm).astype("float64")

        return vw_ma_fast, vw_ma_slow, vpc, vpr, vm, vpci

    def _price_fun(
        self,
        data: PriceDataFrame,
        vpc: pd.Series,
        vpr: pd.Series,
        vpci: pd.Series
    ) -> NDArray[np.float64]:
        """
        Computes a dynamic price adjustment factor based on volume-price interaction.

        This function transforms the selected price series (low or high) using:
        - A dynamic window length (lenV) based on VPCI
        - A corrected volume-price coefficient (VPCc)
        - A rolling average of price / (VPCc * VPR)

        Returns:
            np.ndarray: Adjusted price influence array (price_v), same length as input.
        """
        price_np = self._get_price_series(data).to_numpy()
        vpc_np = vpc.astype("float64").to_numpy()
        vpr_np = vpr.astype("float64").to_numpy()
        vpci_np = vpci.astype("float64").to_numpy()
        lenV = self.compute_len_v(vpc_np, vpci_np) # type: ignore # ????
        VPCc = self.compute_vpcc(vpc_np) # type: ignore # ?????
        return _compute_price_v(price_np, vpr_np, lenV, VPCc) # type: ignore # ?????

    @staticmethod
    def compute_len_v(
        vpc: NDArray[np.float64], 
        vpci: NDArray[np.float64]
    ) -> NDArray[np.int32]:
        """
        Computes a dynamic window length for each time step based on VPCI and VPC.

        - If VPC is negative: window = round(abs(VPCI - 3))
        - If VPC is positive: window = round(VPCI + 3)
        - If VPCI is NaN: fallback to window = 1

        Returns:
            np.ndarray: Array of integer window lengths.
        """
        return np.where(
            np.isnan(vpci), 1,
            np.where(
                vpc < 0,
                np.round(np.abs(vpci - 3)).astype(np.int32),
                np.round(vpci + 3).astype(np.int32)
            )
        )

    @staticmethod
    def compute_vpcc(vpc: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Stabilizes the VPC coefficient to avoid division by near-zero values.

        - If VPC is between -1 and 0 → set to -1.0
        - If VPC is between 0 and 1 → set to 1.0
        - Else → leave unchanged

        Returns:
            np.ndarray: Corrected VPC coefficients.
        """
        return np.where(
            (vpc > -1) & (vpc < 0), -1.0,
            np.where((vpc >= 0) & (vpc < 1), 1.0, vpc)
        )

    def get_last_signal(
        self, 
        data: PriceDataFrame, 
        config: AvslConfigDM
    ) -> float | None:
        """
        Retrieves the most recent value of the computed 
        adaptive level (support or resistance).

        This method is typically used for signal generation, decision-making, or 
        triggering alerts
        based on the latest indicator value. It performs a full AVS calculation and 
        extracts
        the final value from the resulting time series.

                Workflow:
        1. Calls `self.calculate()` to compute the full AVS series (AVSL or AVSR).
        2. Checks if the resulting DataFrame is empty — returns None if no data.
        3. Extracts the last value from the output column (e.g., "avsl" or "avsr").
        4. Returns None if the last value is NaN, otherwise returns it as a float.

        Args:
            data (PriceDataFrame): Historical market data including close, low/high, 
            and volume.
            config (AvslConfigDM): Configuration parameters for AVS calculation.

        Returns:
            float | None: The most recent AVS value, or None if unavailable or invalid.
        """
        df = self.calculate(data, config)
        if df.empty:
            return None
        last_value = df[self._output_column_name()].iloc[-1]
        return None if pd.isna(last_value) else float(last_value)


@njit
def _compute_price_v(
    price: NDArray[np.float64],
    vpr: NDArray[np.float64],
    lenV: NDArray[np.int32],
    VPCc: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Computes a rolling average of price / (VPCc * VPR) over a dynamic window.

    For each time step:
    - Uses a backward-looking window of length `lenV[i]`
    - Applies element-wise division of price by (VPCc * VPR)
    - Averages the result and scales down by 100

    This function is JIT-compiled with Numba for performance.

    Returns:
        np.ndarray: Adjusted price influence array (price_v)
    """
    n = price.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        L = lenV[i]
        if L > 0:
            start = max(0, i - L + 1)
            denom = VPCc[i] * vpr[start:i + 1]
            valid = (VPCc[i] != 0) & (vpr[start:i + 1] != 0)
            values = np.divide(
                price[start:i + 1],
                denom,
                out=np.zeros_like(price[start:i + 1]),
                where=valid
            )
            out[i] = np.sum(values) / L / 100.0
        else:
            out[i] = price[i]
    return out


class AVSL(BaseAVS):
    """
    Adaptive Volume-Weighted Support Level (AVSL) Indicator.

    This indicator calculates a dynamic support level based on volume-weighted price 
    behavior, volume momentum, and volatility. It is designed to adapt to changing 
    market conditions by incorporating volume-sensitive price deviations and 
    smoothing mechanisms.

    The support level is derived from the low price series, adjusted downward by volume
    pressure (via `price_v`) and softened upward by a volatility-based deviation.

    Final formula:
        support = low_price - price_v + deviation

    Where:
        - `low_price` is the raw low price series
        - `price_v` is a dynamic adjustment based on volume-price interaction
        - `deviation` is a volatility buffer derived from VPCI and volume momentum
    """

    def _get_price_series(self, data: PriceDataFrame) -> pd.Series:
        """
        Selects the low price series as the base for support level calculation.

        Args:
            data (PriceDataFrame): Market data containing low, high, close, and volume.

        Returns:
            pd.Series: The low price series.
        """
        return data.low_prices

    def _output_column_name(self) -> str:
        """
        Specifies the name of the output column for the AVSL indicator.

        Returns:
            str: The column name "avsl".
        """
        return "avsl"

    def _adjust_formula(
        self,
        price_series: pd.Series,
        price_v: NDArray[np.float64],
        deviation: pd.Series
    ) -> pd.Series:
        """
        Applies directional logic to compute the adjusted support level.

        The formula subtracts the volume-driven adjustment from the low price,
        then adds a volatility buffer to avoid overly tight support zones.

        Args:
            price_series (pd.Series): The low price series.
            price_v (NDArray[np.float64]): Volume-adjusted price influence.
            deviation (pd.Series): Volatility-based deviation buffer.

        Returns:
            pd.Series: The adjusted support level series.
        """
        return price_series - price_v + deviation


class AVSR(BaseAVS):
    """
    Adaptive Volume-Weighted Resistance Level (AVSR) Indicator.

    This indicator calculates a dynamic resistance level based on volume-weighted 
    price behavior, volume momentum, and volatility. It adapts to market conditions 
    by incorporating volume-sensitive
    price deviations and smoothing mechanisms.

    The resistance level is derived from the high price series, adjusted upward by 
    volume pressure (via `price_v`) and softened downward by a 
    volatility-based deviation.

    Final formula:
        resistance = high_price + price_v - deviation

    Where:
        - `high_price` is the raw high price series
        - `price_v` is a dynamic adjustment based on volume-price interaction
        - `deviation` is a volatility buffer derived from VPCI and volume momentum
    """

    def _get_price_series(self, data: PriceDataFrame) -> pd.Series:
        """
        Selects the high price series as the base for resistance level calculation.

        Args:
            data (PriceDataFrame): Market data containing low, high, close, and volume.

        Returns:
            pd.Series: The high price series.
        """
        return data.high_prices

    def _output_column_name(self) -> str:
        """
        Specifies the name of the output column for the AVSR indicator.

        Returns:
            str: The column name "avsr".
        """
        return "avsr"

    def _adjust_formula(
        self,
        price_series: pd.Series,
        price_v: NDArray[np.float64],
        deviation: pd.Series
    ) -> pd.Series:
        """
        Applies directional logic to compute the adjusted resistance level.

        The formula adds the volume-driven adjustment to the high price,
        then subtracts a volatility buffer to avoid overly optimistic resistance zones.

        Args:
            price_series (pd.Series): The high price series.
            price_v (NDArray[np.float64]): Volume-adjusted price influence.
            deviation (pd.Series): Volatility-based deviation buffer.

        Returns:
            pd.Series: The adjusted resistance level series.
        """
        return price_series + price_v - deviation
