@njit(
    (
        types.float64[:], types.float64[:], types.float64[:], types.float64[:],
        types.float64, types.float64, types.boolean, types.boolean
    ),
    nopython=True,
    cache=True,
    fastmath=True,
)
def _cdl_engulfing_nb(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_body_factor: float,
    max_shadow_factor: float,
    strict: bool,
    symmetric: bool,  # kept for API consistency
) -> np.ndarray:
    """
    Optimized Engulfing pattern.

    Returns:
        1.0 → bullish engulfing
       -1.0 → bearish engulfing
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        o1 = open_[i - 1]
        c1 = close[i - 1]
        h1 = high[i - 1]
        l1 = low[i - 1]

        o0 = open_[i]
        c0 = close[i]
        h0 = high[i]
        l0 = low[i]

        r1 = h1 - l1
        r0 = h0 - l0
        if r1 <= 0.0 or r0 <= 0.0:
            continue

        # Bodies
        b1 = c1 - o1 if c1 > o1 else o1 - c1
        b0 = c0 - o0 if c0 > o0 else o0 - c0
        if b1 <= 0.0 or b0 <= 0.0:
            continue

        # Colors
        bull1 = c1 > o1
        bear1 = c1 < o1
        bull0 = c0 > o0
        bear0 = c0 < o0

        direction = 0.0

        # Bullish engulfing
        if bear1 and bull0:
            if o0 <= c1 and c0 >= o1:
                direction = 1.0

        # Bearish engulfing
        elif bull1 and bear0:
            if o0 >= c1 and c0 <= o1:
                direction = -1.0

        if direction == 0.0:
            continue

        if strict:
            # Body size requirement
            if b0 < min_body_factor * r0 or b1 < min_body_factor * r1:
                continue

            # Shadows (fast)
            up1 = o1 if o1 > c1 else c1
            lo1 = c1 if o1 > c1 else o1
            sh1 = (h1 - up1) + (lo1 - l1)

            up0 = o0 if o0 > c0 else c0
            lo0 = c0 if o0 > c0 else o0
            sh0 = (h0 - up0) + (lo0 - l0)

            if sh1 > max_shadow_factor * r1 or sh0 > max_shadow_factor * r0:
                continue

        out[i] = direction

    return out


def cdl_engulfing(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    strict: bool = False,
    symmetric: bool = False,
    min_body_factor: float = 0.3,
    max_shadow_factor: float = 0.5,
) -> np.ndarray:
    """
    Engulfing pattern with strict support.
    """
    if isinstance(open_, pl.Series): open_ = open_.to_numpy()
    if isinstance(high, pl.Series): high = high.to_numpy()
    if isinstance(low, pl.Series): low = low.to_numpy()
    if isinstance(close, pl.Series): close = close.to_numpy()

    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if use_talib and talib_available and not symmetric:
        talib_out = talib.CDLENGULFING(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    out = _cdl_engulfing_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric,
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_engulfing_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    strict: bool = False,
    symmetric: bool = False,
    min_body_factor: float = 0.3,
    max_shadow_factor: float = 0.5,
    output_col: str = "CDL_ENGULFING",
) -> pl.DataFrame:
    out = cdl_engulfing(
        df[open_col].to_numpy(),
        df[high_col].to_numpy(),
        df[low_col].to_numpy(),
        df[close_col].to_numpy(),
        offset=offset,
        fillna=fillna,
        strict=strict,
        symmetric=symmetric,
        min_body_factor=min_body_factor,
        max_shadow_factor=max_shadow_factor,
    )
    return df.with_columns(pl.Series(output_col, out))
