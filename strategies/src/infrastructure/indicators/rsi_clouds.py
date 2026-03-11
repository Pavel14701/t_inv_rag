import numpy as np
import polars as pl


# ----------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------
def rsi_clouds_signals_numpy(
    macd_line: np.ndarray,
    macd_signal_line: np.ndarray,
    macd_hist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate buy/sell signals based on MACD crossovers.

    Returns
    -------
    (macd_cross_signal, hist_cross_zero)
    """
    n = len(macd_line)
    macd_cross = np.zeros(n, dtype=int)
    hist_cross = np.zeros(n, dtype=int)
    # Ручной сдвиг для предыдущих значений (без копирования всего массива)
    prev_macd = np.empty_like(macd_line)
    prev_macd[0] = np.nan
    prev_macd[1:] = macd_line[:-1]
    prev_signal = np.empty_like(macd_signal_line)
    prev_signal[0] = np.nan
    prev_signal[1:] = macd_signal_line[:-1]
    # MACD line crosses signal
    buy_mask = (prev_macd < prev_signal) & (macd_line > macd_signal_line)
    sell_mask = (prev_macd > prev_signal) & (macd_line < macd_signal_line)
    macd_cross[buy_mask] = 1
    macd_cross[sell_mask] = -1
    # Histogram crosses zero
    prev_hist = np.empty_like(macd_hist)
    prev_hist[0] = np.nan
    prev_hist[1:] = macd_hist[:-1]
    up_mask = (prev_hist < 0) & (macd_hist > 0)
    down_mask = (prev_hist > 0) & (macd_hist < 0)
    hist_cross[up_mask] = 1
    hist_cross[down_mask] = -1
    return macd_cross, hist_cross


def get_last_rsi_clouds_signal(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    rsi_length: int = 14,
    rsi_scalar: float = 100.0,
    rsi_drift: int = 1,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    macd_mamode: str = "ema",
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    nan_policy: str = "raise",
    trim: bool = False,
) -> str | None:
    """
    Get last MACD crossover signal: 'buy', 'sell' or None.
    """
    rsi, macd_line, macd_signal_line, macd_hist = rsi_clouds_ind(
        open_,
        high,
        low,
        close,
        rsi_length=rsi_length,
        rsi_scalar=rsi_scalar,
        rsi_drift=rsi_drift,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        macd_mamode=macd_mamode,
        offset=offset,
        fillna=fillna,
        use_talib=use_talib,
        nan_policy=nan_policy,
        trim=trim,
    )
    if len(macd_line) == 0:
        return None
    macd_cross, _ = rsi_clouds_signals_numpy(macd_line, macd_signal_line, macd_hist)
    last = macd_cross[-1]
    if last == 1:
        return "buy"
    if last == -1:
        return "sell"
    return None
