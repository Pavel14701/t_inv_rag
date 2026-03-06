import numpy as np
import polars as pl


# ----------------------------------------------------------------------
# Signal generation functions
# ----------------------------------------------------------------------
def generate_scrsi_signals_numpy(
    crsi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate buy/sell signals based on SCRSI levels.

    Parameters
    ----------
    crsi : np.ndarray
        Smoothed SCRSI values.

    Returns
    -------
    (buy, sell) arrays of int (0/1) with same length as input.
    """
    n = len(crsi)
    buy = np.zeros(n, dtype=int)
    sell = np.zeros(n, dtype=int)
    crsi_shift = np.roll(crsi, 1)
    crsi_shift[0] = np.nan
    # Buy signals
    buy_mask = (crsi_shift < 50) & (crsi >= 50)
    buy_mask |= (crsi_shift <= 0) & (crsi > 0)
    buy[buy_mask] = 1
    # Sell signals
    sell_mask = (crsi_shift > 50) & (crsi <= 50)
    sell_mask |= (crsi_shift >= 100) & (crsi < 100)
    sell[sell_mask] = 1
    return buy, sell


def get_last_scrsi_signal(
    close: np.ndarray | pl.Series,
    domcycle: int,
    vibration: int,
    leveling: float,
    nan_policy: str = 'raise',
    trim: bool = False,
    offset: int = 0,
    fillna: float | None = None,
) -> str | None:
    """
    Determine last trading signal (long/short) from SCRSI.

    Returns
    -------
    'long', 'short', or None.
    """
    # Рассчитываем SCRSI (нам нужен только crsi)
    _, crsi, _, _ = scrsi_ind(
        close,
        domcycle=domcycle,
        vibration=vibration,
        leveling=leveling,
        nan_policy=nan_policy,
        trim=trim,
        offset=offset,
        fillna=fillna,
    )
    if len(crsi) == 0:
        return None
    buy, sell = generate_scrsi_signals_numpy(crsi)
    if buy[-1] == 1:
        return 'long'
    if sell[-1] == 1:
        return 'short'
    return None