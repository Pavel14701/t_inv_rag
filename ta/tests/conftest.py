"""Shared fixtures for the TA test suite.

Provides synthetic price series that mimic real market data:
- random walk with drift, noise, and optional trend
- predefined patterns: uptrend, downtrend, sideways, with reversals
- Polars DataFrames for integration tests
"""

import pytest
import numpy as np
import polars as pl

# -----------------------------------------------------------------------------
# Constants for synthetic data generation
# -----------------------------------------------------------------------------

# Default parameters for random walk
DEFAULT_N = 200
DEFAULT_START_PRICE = 100.0
DEFAULT_DRIFT = 0.1
DEFAULT_VOLATILITY = 1.0
DEFAULT_NOISE_SCALE = 0.5
DEFAULT_SEED = 42

# Parameters for specific patterns
PATTERN_UPTREND_START = 100.0
PATTERN_UPTREND_END = 200.0
PATTERN_UPTREND_LEN = 50

PATTERN_DOWNDTREND_START = 200.0
PATTERN_DOWNDTREND_END = 100.0
PATTERN_DOWNDTREND_LEN = 50

PATTERN_SIDEWAYS_CENTER = 100.0
PATTERN_SIDEWAYS_AMPLITUDE = 10.0
PATTERN_SIDEWAYS_LEN = 100

PATTERN_UP_DOWN_START = 100.0
PATTERN_UP_DOWN_PEAK = 160.0
PATTERN_UP_DOWN_END = 120.0
PATTERN_UP_DOWN_SEG_LEN = 30

PATTERN_DOWN_UP_START = 120.0
PATTERN_DOWN_UP_BOTTOM = 80.0
PATTERN_DOWN_UP_END = 140.0
PATTERN_DOWN_UP_SEG_LEN = 30

PATTERN_REVERSAL_SEG_LEN = 40
PATTERN_REVERSAL_START = 100.0
PATTERN_REVERSAL_MID = 150.0
PATTERN_REVERSAL_BOTTOM = 120.0
PATTERN_REVERSAL_END = 180.0

# Random walk variations
VOLATILE_VOLATILITY = 3.0
VOLATILE_SEED = 123

# OHLC DataFrame parameters
OHLC_NOISE_SCALE = 0.5
OHLC_HIGH_LOW_SCALE = 0.8
OHLC_SEED = 42

# Box size and reversal for Renko/P&F (used in other tests)
DEFAULT_BOX_SIZE = 2.0
DEFAULT_REVERSAL = 3

# Offset and fillna defaults for tests
DEFAULT_OFFSET = 0
DEFAULT_FILLNA = 0.0


# -----------------------------------------------------------------------------
# Synthetic price generators
# -----------------------------------------------------------------------------

def generate_random_walk(
    n: int = DEFAULT_N,
    start_price: float = DEFAULT_START_PRICE,
    drift: float = DEFAULT_DRIFT,
    volatility: float = DEFAULT_VOLATILITY,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    seed: int | None = DEFAULT_SEED,
) -> np.ndarray:
    """Generate a synthetic price series (random walk with drift and noise).

    Parameters
    ----------
    n : int
        Number of bars.
    start_price : float
        Initial price.
    drift : float
        Trend per step (positive = uptrend, negative = downtrend).
    volatility : float
        Scale of the random step.
    noise_scale : float
        Additional independent noise added to each price.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        1D float64 array of prices.

    """
    if seed is not None:
        np.random.seed(seed)
    steps = np.random.randn(n) * volatility + drift
    prices = start_price + np.cumsum(steps)
    if noise_scale > 0:
        prices += np.random.randn(n) * noise_scale
    return prices.astype(np.float64)


def generate_pattern(
    pattern: str,
    n: int = 50,
    start_price: float = 100.0,
    box_size: float = DEFAULT_BOX_SIZE,
) -> np.ndarray:
    """Generate a predefined price pattern for testing.

    Patterns:
    - 'uptrend': steady increase by box_size each step.
    - 'downtrend': steady decrease.
    - 'sideways': price oscillates within a range.
    - 'up_then_down': increase then decrease.
    - 'down_then_up': decrease then increase.
    - 'volatile': random walk with high volatility.

    Returns
    -------
    np.ndarray
        1D float64 array of prices.

    """
    if pattern == 'uptrend':
        return np.linspace(start_price, start_price + box_size * n, n)
    elif pattern == 'downtrend':
        return np.linspace(start_price, start_price - box_size * n, n)
    elif pattern == 'sideways':
        return start_price + box_size * 0.5 * np.sin(
            np.linspace(0, 4 * np.pi, n)
        )
    elif pattern == 'up_then_down':
        half = n // 2
        up = np.linspace(start_price, start_price + box_size * half, half)
        down = np.linspace(up[-1], up[-1] - box_size * (n - half), n - half)
        return np.concatenate([up, down])
    elif pattern == 'down_then_up':
        half = n // 2
        down = np.linspace(start_price, start_price - box_size * half, half)
        up = np.linspace(down[-1], down[-1] + box_size * (n - half), n - half)
        return np.concatenate([down, up])
    elif pattern == 'volatile':
        return generate_random_walk(
            n, start_price, drift=0,
            volatility=box_size * 1.5,
            seed=DEFAULT_SEED
        )
    else:
        raise ValueError(f'Unknown pattern: {pattern}')


# -----------------------------------------------------------------------------
# Fixtures for price series
# -----------------------------------------------------------------------------

@pytest.fixture
def prices_uptrend() -> np.ndarray:
    """Simple uptrend: linear increase."""
    return np.linspace(
        PATTERN_UPTREND_START,
        PATTERN_UPTREND_END,
        PATTERN_UPTREND_LEN
    )


@pytest.fixture
def prices_downtrend() -> np.ndarray:
    """Simple downtrend: linear decrease."""
    return np.linspace(
        PATTERN_DOWNDTREND_START,
        PATTERN_DOWNDTREND_END,
        PATTERN_DOWNDTREND_LEN
    )


@pytest.fixture
def prices_sideways() -> np.ndarray:
    """Sideways market: oscillation within a range."""
    return PATTERN_SIDEWAYS_CENTER + PATTERN_SIDEWAYS_AMPLITUDE * np.sin(
        np.linspace(0, 4 * np.pi, PATTERN_SIDEWAYS_LEN)
    )


@pytest.fixture
def prices_up_then_down() -> np.ndarray:
    """Price rises then falls."""
    up = np.linspace(
        PATTERN_UP_DOWN_START,
        PATTERN_UP_DOWN_PEAK,
        PATTERN_UP_DOWN_SEG_LEN
    )
    down = np.linspace(
        PATTERN_UP_DOWN_PEAK,
        PATTERN_UP_DOWN_END,
        PATTERN_UP_DOWN_SEG_LEN
    )
    return np.concatenate([up, down])


@pytest.fixture
def prices_down_then_up() -> np.ndarray:
    """Price falls then rises."""
    down = np.linspace(
        PATTERN_DOWN_UP_START,
        PATTERN_DOWN_UP_BOTTOM,
        PATTERN_DOWN_UP_SEG_LEN
    )
    up = np.linspace(
        PATTERN_DOWN_UP_BOTTOM,
        PATTERN_DOWN_UP_END,
        PATTERN_DOWN_UP_SEG_LEN
    )
    return np.concatenate([down, up])


@pytest.fixture
def prices_random_walk() -> np.ndarray:
    """Random walk with moderate volatility."""
    return generate_random_walk(
        n=DEFAULT_N,
        start_price=DEFAULT_START_PRICE,
        drift=DEFAULT_DRIFT,
        volatility=DEFAULT_VOLATILITY,
        seed=DEFAULT_SEED
    )


@pytest.fixture
def prices_volatile() -> np.ndarray:
    """High volatility random walk."""
    return generate_random_walk(
        n=DEFAULT_N,
        start_price=DEFAULT_START_PRICE,
        drift=0,
        volatility=VOLATILE_VOLATILITY,
        seed=VOLATILE_SEED
    )


@pytest.fixture
def prices_with_reversals() -> np.ndarray:
    """Price series with clear trend reversals."""
    seg1 = np.linspace(
        PATTERN_REVERSAL_START,
        PATTERN_REVERSAL_MID,
        PATTERN_REVERSAL_SEG_LEN
    )
    seg2 = np.linspace(
        PATTERN_REVERSAL_MID,
        PATTERN_REVERSAL_BOTTOM,
        PATTERN_REVERSAL_SEG_LEN
    )
    seg3 = np.linspace(
        PATTERN_REVERSAL_BOTTOM,
        PATTERN_REVERSAL_END,
        PATTERN_REVERSAL_SEG_LEN
    )
    return np.concatenate([seg1, seg2, seg3])


@pytest.fixture
def df_ohlc(prices_random_walk: np.ndarray) -> pl.DataFrame:
    """DataFrame with open, high, low, close columns."""  # noqa: D403
    np.random.seed(OHLC_SEED)
    n = len(prices_random_walk)
    return pl.DataFrame({
        'date': np.arange(n),
        'open': prices_random_walk + np.random.randn(n) * OHLC_NOISE_SCALE,
        'high': prices_random_walk + np.abs(
            np.random.randn(n) * OHLC_HIGH_LOW_SCALE
        ),
        'low': prices_random_walk - np.abs(
            np.random.randn(n) * OHLC_HIGH_LOW_SCALE
        ),
        'close': prices_random_walk,
    })


# -----------------------------------------------------------------------------
# Fixtures for Polars DataFrames
# -----------------------------------------------------------------------------

@pytest.fixture
def df_uptrend(prices_uptrend) -> pl.DataFrame:  # noqa: D103
    return pl.DataFrame({'close': prices_uptrend})


@pytest.fixture
def df_downtrend(prices_downtrend) -> pl.DataFrame:  # noqa: D103
    return pl.DataFrame({'close': prices_downtrend})


@pytest.fixture
def df_sideways(prices_sideways) -> pl.DataFrame:  # noqa: D103
    return pl.DataFrame({'close': prices_sideways})


@pytest.fixture
def df_random_walk(prices_random_walk) -> pl.DataFrame:  # noqa: D103
    return pl.DataFrame({'close': prices_random_walk})


@pytest.fixture
def df_volatile(prices_volatile) -> pl.DataFrame:  # noqa: D103
    return pl.DataFrame({'close': prices_volatile})


# -----------------------------------------------------------------------------
# Fixtures for common parameters
# -----------------------------------------------------------------------------

@pytest.fixture
def box_size() -> float:
    """Default box size for Renko and P&F tests."""
    return DEFAULT_BOX_SIZE


@pytest.fixture
def reversal() -> int:
    """Default reversal size for P&F tests."""
    return DEFAULT_REVERSAL


@pytest.fixture
def offset() -> int:
    """Default offset for shift tests."""
    return DEFAULT_OFFSET


@pytest.fixture
def fillna() -> float:
    """Default fillna value."""
    return DEFAULT_FILLNA


# -----------------------------------------------------------------------------
# Fixtures for IEEE 754 edge cases (NaN, Inf, extreme, empty, short, etc.)
# -----------------------------------------------------------------------------

@pytest.fixture
def prices_with_nan() -> np.ndarray:
    """Price series with a NaN at index 5 (length 15)."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)


@pytest.fixture
def prices_with_inf() -> np.ndarray:
    """Price series with an Inf at index 5 (length 15)."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.inf, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0], dtype=np.float64)


@pytest.fixture
def prices_extreme() -> np.ndarray:
    """Price series with extreme values (1e300 and 1e-300) (length 15)."""
    return np.array([1e300, 1e-300, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 13.0], dtype=np.float64)


@pytest.fixture
def prices_all_nan() -> np.ndarray:
    """All NaN prices (length 10)."""
    return np.full(10, np.nan, dtype=np.float64)


@pytest.fixture
def prices_empty() -> np.ndarray:
    """Empty array."""
    return np.array([], dtype=np.float64)


@pytest.fixture
def prices_short() -> np.ndarray:
    """Short series (length 3)."""
    return np.array([1.0, 2.0, 3.0], dtype=np.float64)


@pytest.fixture
def prices_single() -> np.ndarray:
    """Single element (length 1)."""
    return np.array([10.0], dtype=np.float64)
