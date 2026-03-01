from dataclasses import dataclass
from typing import Any, Tuple, cast

import numpy as np
import pandas as pd # type: ignore
import pandas_ta as ta # type: ignore
from numpy.typing import NDArray
from scipy.signal import find_peaks # type: ignore

from domain.entities import OrderBlockDetectorDM
from infrastructure._types import PriceDataFrame


@dataclass
class OrderBlock:
    block_type: str
    start: pd.Timestamp
    break_: pd.Timestamp
    retest: pd.Timestamp
    zone_low: float
    zone_high: float


class OrderBlockDetector:
    """
    Detects supply and demand zones (order blocks) using ZigZag peaks/valleys,
    breakout logic, volume confirmation, ATR sizing, and liquidity clustering.
    """

    def identify_order_blocks(
        self,
        data: PriceDataFrame,
        config: OrderBlockDetectorDM,
        lookback: int = 20,
        confirmation_window: int = 10,
        min_reaction_size: float = 0.002,
        atr_period: int = 14,
        volume_window: int = 20,
        liquidity_window: int = 10,
        liquidity_tolerance: float = 0.001
    ) -> pd.DataFrame:
        """
        Full pipeline for detecting and confirming order blocks.

        Args:
            data: Price data with OHLCV.
            config: ZigZag detection config.
            lookback: Candles between peak/valley and breakout.
            confirmation_window: Candles to wait for retest.
            min_reaction_size: Minimum % move after retest.
            atr_period: ATR window for zone sizing.
            volume_window: Rolling window for volume.
            liquidity_window: Rolling window for local highs/lows.
            liquidity_tolerance: Max distance to consider liquidity proximity.

        Returns:
            pd.DataFrame of confirmed blocks.
        """
        zigzag_df = self._zigzag_indicator(data, config)
        indicators = self._precompute_indicators(
            data, 
            atr_period, 
            volume_window, 
            liquidity_window
        )
        candidates = self._generate_block_candidates(
            data, 
            zigzag_df, 
            indicators, 
            lookback, 
            liquidity_tolerance
        )
        confirmed = self._validate_block_candidates(
            data, 
            candidates, 
            indicators, 
            confirmation_window, 
            min_reaction_size
        )
        return pd.DataFrame([b.__dict__ for b in confirmed])

    def _zigzag_indicator(
        self, 
        data: PriceDataFrame, 
        config: OrderBlockDetectorDM
    ) -> pd.DataFrame:
        """
        Detects local peaks and valleys using ZigZag logic via scipy.signal.find_peaks.

        Args:
            data: PriceDataFrame with high and low price series.
            config: Configuration object with peak detection parameters.

        Returns:
            pd.DataFrame with two columns:
                - peaks: high price values at detected peaks, NaN elsewhere
                - valleys: low price values at detected valleys, NaN elsewhere
        """
        peaks = self._detect_peaks(data.high_prices, config, is_peak=True)
        valleys = self._detect_peaks(data.low_prices, config, is_peak=False)
        return pd.DataFrame({
            "peaks": self._mark_extremes(data.high_prices, peaks),
            "valleys": self._mark_extremes(data.low_prices, valleys)
        }, index=data.index)

    def _precompute_indicators(
        self, 
        data: PriceDataFrame, 
        atr_period: int, 
        volume_window: int, 
        liquidity_window: int
    ) -> dict:
        """
        Computes supporting indicators used for zone sizing and validation.

        Returns:
            Dictionary with:
                - atr: Average True Range series
                - avg_volume: rolling average volume
                - local_highs: rolling max of high prices
                - local_lows: rolling min of low prices
                - zone_low: lower bound of zone (close - ATR)
                - zone_high: upper bound of zone (close + ATR)
        """
        atr = self._calculate_atr(data, atr_period)
        return {
            "atr": atr,
            "avg_volume": data.volume.rolling(window=volume_window).mean(),
            "local_highs": data.high_prices.rolling(window=liquidity_window).max(),
            "local_lows": data.low_prices.rolling(window=liquidity_window).min(),
            "zone_low": data.close_prices - atr,
            "zone_high": data.close_prices + atr
        }

    def _generate_block_candidates(
        self, 
        data: PriceDataFrame, 
        zigzag_df: pd.DataFrame, 
        indicators: dict, 
        lookback: int, 
        liquidity_tolerance: float
    ) -> list[dict]:
        """
        Generates potential order block candidates based on ZigZag extrema and 
        liquidity proximity.

        Returns:
            List of candidate dictionaries with:
                - type: "supply" or "demand"
                - idx: index of peak/valley
                - break_idx: index of breakout
        """
        candidates = []
        for i in range(lookback, len(data)):
            idx = i - lookback
            if not np.isnan(zigzag_df.peaks.iloc[idx]):
                if self._has_liquidity_cluster(
                    data.low_prices, 
                    idx, 
                    indicators["local_lows"], 
                    liquidity_tolerance
                ):
                    candidates.append(
                        {
                            "block_type": "supply", 
                            "idx": idx, 
                            "break_idx": i
                        }
                    )
            elif not np.isnan(zigzag_df.valleys.iloc[idx]):
                if self._has_liquidity_cluster(
                    data.high_prices, 
                    idx, 
                    indicators["local_highs"], 
                    liquidity_tolerance
                ):
                    candidates.append(
                        {
                            "block_type": "demand", 
                            "idx": idx, 
                            "break_idx": i
                        }
                    )
        return candidates

    def _validate_block_candidates(
        self, 
        data: PriceDataFrame, 
        candidates: list[dict], 
        indicators: dict, 
        confirmation_window: int, 
        min_reaction_size: float
    ) -> list[OrderBlock]:
        """
        Validates each candidate by checking breakout, retest, and reaction strength.

        Returns:
            List of confirmed block dictionaries with full zone metadata.
        """
        confirmed = []
        for c in candidates:
            idx = c["idx"]
            breakout_idx = c["break_idx"]
            is_supply = c["block_type"] == "supply"
            direction = "down" if is_supply else "up"

            if not self._is_valid_breakout(
                data, 
                idx, 
                breakout_idx, 
                indicators["avg_volume"], 
                direction
            ):
                continue

            block = self._confirm_block(
                data=data,
                idx=idx,
                breakout_idx=breakout_idx,
                window=confirmation_window,
                zone_low=indicators["zone_low"].iloc[idx],
                zone_high=indicators["zone_high"].iloc[idx],
                avg_volume=indicators["avg_volume"],
                min_reaction_size=min_reaction_size,
                is_supply=is_supply,
            )
            if block:
                confirmed.append(block)
        return confirmed

    def _confirm_block(
        self, 
        data: PriceDataFrame, 
        idx: int, 
        breakout_idx: int, 
        window: int, 
        zone_low: float, 
        zone_high: float, 
        avg_volume: pd.Series, 
        min_reaction_size: float, 
        is_supply: bool
    ) -> OrderBlock | None:
        """
        Confirms a block by checking for valid retest and reaction within the zone.

        Args:
            idx: Index of peak/valley.
            breakout_idx: Index of breakout.
            window: Number of candles to wait for retest.
            zone_low: Lower bound of zone.
            zone_high: Upper bound of zone.
            avg_volume: Rolling average volume series.
            min_reaction_size: Minimum percentage move to confirm reaction.
            is_supply: True for supply block, False for demand block.

        Returns:
            dict with block metadata if confirmed, else None.
        """
        test_series = data.low_prices if is_supply else data.high_prices
        close = data.close_prices
        volume = data.volume
        block_type = "supply" if is_supply else "demand"

        end_idx = min(breakout_idx + window, len(data))
        for j in range(breakout_idx + 1, end_idx):
            if not (
                zone_low <= (test_series.iloc[j]) <= zone_high
                and (volume.iloc[j]) > avg_volume.iloc[j]
            ):
                continue

            close_j = close.iloc[j]
            # Reaction is measured as % move beyond the zone boundary:
            reaction = ((zone_low - close_j) / zone_low) if is_supply else (
                (close_j - zone_high) / zone_high
            )
            if reaction >= min_reaction_size:
                return OrderBlock(
                    block_type=block_type,
                    start=data.index[idx],
                    break_=data.index[breakout_idx],
                    retest=data.index[j],
                    zone_low=zone_low,
                    zone_high=zone_high
                )
        return None

    def _detect_peaks(
        self, 
        series: pd.Series, 
        config: OrderBlockDetectorDM, 
        is_peak: bool
    ) -> NDArray[np.intp]:
        """
        Applies scipy.signal.find_peaks to detect local extrema in a price series.

        Args:
            series: The price series to analyze (typically high or low prices).
            config: Configuration object containing peak detection parameters.
            is_peak: If True, detects peaks; if False, detects valleys.

        Returns:
            Array of integer indices where peaks or valleys were detected.
        """
        prominence = config.peak_prominance if is_peak else config.valley_prominance
        result = find_peaks(
            x=series,
            height=config.height,
            threshold=config.threshold,
            distance=config.distance,
            prominence=prominence,
            width=config.width,
            wlen=config.wlen,
            rel_height=config.rel_height,
            plateau_size=config.plateu_size
        )
        return cast(Tuple[NDArray[np.intp], dict[str, Any]], result)[0]

    def _mark_extremes(
        self, 
        series: pd.Series, 
        indices: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        """
        Creates a NaN-filled array with values only at specified indices.

        Args:
            series: Original price series.
            indices: Indices of detected peaks or valleys.

        Returns:
            NumPy array with NaNs everywhere except at the specified indices,
            where the original price values are retained.
        """
        arr = np.full_like(series.to_numpy(), np.nan, dtype=np.float64)
        arr[indices] = series.iloc[indices]
        return arr

    def _calculate_atr(
        self, 
        data: PriceDataFrame, 
        period: int
    ) -> pd.Series:
        """
        Calculates the Average True Range (ATR) using pandas_ta with TA-Lib backend.

        ATR is a volatility indicator that measures the average range between
        high and low prices over a specified period, accounting for gaps.

        Args:
            data: Price data containing high, low, and close prices.
            period: Number of periods to use for the ATR calculation.

        Returns:
            A pandas Series containing the ATR values.
        """
        return ta.atr( # type: ignore
            high=data.high_prices,
            low=data.low_prices,
            close=data.close_prices,
            length=period,
            talib=True
        )

    def _is_valid_breakout(
        self, 
        data: PriceDataFrame, 
        idx: int, 
        i: int, 
        avg_volume: pd.Series, 
        direction: str
    ) -> bool:
        """
        Determines whether a breakout is valid based on price movement and volume.

        Args:
            data: Price data.
            idx: Index of the peak or valley.
            i: Index of the breakout candle.
            avg_volume: Rolling average volume series.
            direction: "up" for demand zones, "down" for supply zones.

        Returns:
            True if the breakout is valid; otherwise, False.
        """
        if direction == "down":
            return data.low_prices.iloc[i] < data.low_prices.iloc[idx] \
                and data.volume.iloc[i] > avg_volume.iloc[i]
        else:
            return data.high_prices.iloc[i] > data.high_prices.iloc[idx] \
                and data.volume.iloc[i] > avg_volume.iloc[i]

    def _has_liquidity_cluster(
        self, 
        series: pd.Series, 
        idx: int, 
        local_extremes: pd.Series, 
        tolerance: float
    ) -> bool:
        """
        Checks whether a price point is near a local high/low, indicating a 
        liquidity cluster.

        Args:
            series: The price series (high or low).
            idx: Index to evaluate.
            local_extremes: Rolling max/min series representing local liquidity zones.
            tolerance: Maximum allowed distance to consider proximity.

        Returns:
            True if the price is within the tolerance of a local extreme.
        """
        return abs(local_extremes.iloc[idx] - series.iloc[idx]) < tolerance
