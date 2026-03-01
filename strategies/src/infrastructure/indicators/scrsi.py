import numpy as np
import pandas as pd # type: ignore
from numpy.typing import NDArray

from domain.entities import ScrsiConfigDM
from infrastructure._types import PriceDataFrame


class SmoothCicleRsi:
    """
    SmoothCicle RSI (SCRSI) Indicator Implementation.

    This indicator calculates an adaptive RSI that includes 
    smoothing techniques to filter noise.
    It measures price momentum based on the 
    difference between closing prices over a 
    set cycle length.

    Formula:
    1. Calculate price difference (`diff = close_prices.diff()`).
    2. Compute the average of positive and negative movements:
       - up = average gain over `cyclelen` periods.
       - down = average loss over `cyclelen` periods.
    3. Standard RSI formula: RSI = 100 - (100 / (1 + up/down)).
    4. Convert RSI from range [0,100] to [-100,100] to highlight trends.
    5. Apply cyclic smoothing to refine RSI momentum shifts.
    6. Define boundaries (`Lower Bound` and `Upper Bound`)
       based on percentile thresholds.

    Usage:
    - Provides momentum-based trading signals.
    - Generates BUY/SELL signals based on 
      RSI thresholds (crossing level 50, reversals from 0/100).
    """

    def calculate_scrsi(
        self, 
        data: PriceDataFrame, 
        config: ScrsiConfigDM
    ) -> pd.DataFrame:
        """
        Computes the SmoothCicle RSI (SCRSI) values.
        Args:
            data (PriceDataFrame): The price dataset containing close prices.
            config (ScrsiConfigDM): Indicator configuration (cycle length, 
            vibration, smoothing).
        Returns:
            pd.DataFrame: A DataFrame containing the SCRSI 
            scaled values and boundary levels.
        """        
        # Compute cycle length and cyclic memory for smoothing
        cyclelen = config.domcycle // 2
        cyclicmemory = config.domcycle * 2
        # Calculate price differences
        price_diff = data.close_prices.diff()
        # Separate gains and losses
        up_raw = np.maximum(price_diff, 0)
        down_raw = np.minimum(price_diff, 0)
        # Rolling averages
        up = pd.Series(  # type: ignore
            up_raw, 
            index=data.index,  # type: ignore
            dtype=float
        ).rolling(
            window=cyclelen
        ).mean()  # type: ignore
        down = pd.Series(  # type: ignore
            down_raw, 
            index=data.index,  # type: ignore
            dtype=float
        ).rolling(
            window=cyclelen
        ).mean()  # type: ignore
        # Convert to numpy arrays with explicit dtype
        up_arr: NDArray[np.float64] = up.to_numpy(dtype=np.float64)  # type: ignore
        down_arr: NDArray[np.float64] = down.to_numpy(dtype=np.float64)  # type: ignore
        # Standard RSI calculation with safe division
        rsi = 100 - 100 / (1 + np.divide(
            up_arr, down_arr,
            out=np.zeros_like(up_arr, dtype=np.float64),
            where=down_arr != 0
        ))
        # Clip RSI to [0, 100]
        rsi = np.clip(rsi, 0, 100)
        # Scale RSI to [-100, 100]
        rsi_scaled = (rsi - 50) * 2
        # Apply cyclic smoothing
        torque = 2.0 / (config.vibration + 1)
        phasingLag = (config.vibration - 1) // 2
        crsi = np.zeros_like(rsi_scaled, dtype=np.float64)
        mask = np.arange(len(rsi_scaled)) >= phasingLag
        crsi[mask] = torque * (
            2 * rsi_scaled[mask] - np.roll(rsi_scaled, phasingLag)[mask]
        ) + (1 - torque) * np.roll(crsi, 1)[mask]
        # Clean NaNs before boundary calculation
        crsi_clean = crsi[~np.isnan(crsi)]
        db = float(
            np.percentile(crsi_clean[:cyclicmemory], config.leveling)
            if len(crsi_clean[:cyclicmemory]) > 0 else -100
        )
        ub = float(
            np.percentile(crsi_clean[:cyclicmemory], 100 - config.leveling)
            if len(crsi_clean[:cyclicmemory]) > 0 else 100
        )
        # Flatten arrays if needed
        if getattr(crsi, "ndim", 1) > 1:
            crsi = crsi.flatten()
        if getattr(rsi_scaled, "ndim", 1) > 1:
            rsi_scaled = rsi_scaled.flatten()
        # Return DataFrame
        return pd.DataFrame({
            'CRSI Scaled': rsi_scaled,
            'Lower Bound': db,
            'Upper Bound': ub
        }, index=data.index)  # type: ignore

    def generate_scrsi_signals(
        self, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generates trading signals based on SCRSI levels.
        Signals:
        - Buy: RSI crossing above level 50 or bouncing from 0 (oversold zone).
        - Sell: RSI crossing below level 50 or bouncing from 100 (overbought zone).
        Args:
            data (pd.DataFrame): DataFrame containing SCRSI values.
        Returns:
            pd.DataFrame: A DataFrame with separate buy and sell signal series.
        """
        # Create a series for BUY signals (0 = no signal, 1 = buy)
        buy_signals = pd.Series(0, index=data.index)  # type: ignore
        # Buy signal: Crossing 50 from below
        buy_signals[
            (
                data["CRSI Scaled"].shift(1) < 50  # type: ignore
            ) & (
                data["CRSI Scaled"] >= 50
            )
        ] = 1  
        # Buy signal: Reversal from oversold region (0)
        buy_signals[
            (
                data["CRSI Scaled"].shift(1) <= 0  # type: ignore
            ) & (
                data["CRSI Scaled"] > 0
            )
        ] = 1
        # Create a series for SELL signals (0 = no signal, 1 = sell)
        sell_signals = pd.Series(0, index=data.index)  # type: ignore
        # Sell signal: Crossing 50 from above
        sell_signals[
            (
                data["CRSI Scaled"].shift(1) > 50  # type: ignore
            ) & (
                data["CRSI Scaled"] <= 50
            )
        ] = 1
        # Sell signal: Reversal from overbought region (100)
        sell_signals[
            (
                data["CRSI Scaled"].shift(1) >= 100  # type: ignore
            ) & (
                data["CRSI Scaled"] < 100
            )
        ] = 1  
        return pd.DataFrame(
            {
                "buy_signals": buy_signals, 
                "sell_signals": sell_signals
            }, 
            index=data.index  # type: ignore
        )

    def get_last_signal(
        self, 
        data: PriceDataFrame, 
        config: ScrsiConfigDM
    ) -> str | None:
        """
        Determines the last trading signal based on SCRSI.
        - If the last bar has a buy signal → returns "long".
        - If the last bar has a sell signal → returns "short".
        - If no signal is detected → returns None.
        Args:
            data (PriceDataFrame): Market data.
            config (ScrsiConfigDM): SCRSI indicator configuration.
        Returns:
            str | None: "long" if buy signal, "short" if sell signal, 
            or None if no signal is present.
        """
        # Compute SCRSI values
        scrsi_df = self.calculate_scrsi(data, config)
        # Generate trading signals
        signals_df = self.generate_scrsi_signals(scrsi_df)
        # Check the last signal and return appropriate action
        if signals_df["buy_signals"].iloc[-1] == 1:  # type: ignore
            return "long"
        elif signals_df["sell_signals"].iloc[-1] == 1:  # type: ignore
            return "short"
        return None
