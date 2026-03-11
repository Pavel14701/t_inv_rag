@njit(
    (
        types.float64[:], types.float64[:], types.float64[:], types.float64[:],
        types.float64, types.float64, types.boolean, types.boolean
    ),
    nopython=True,
    cache=True,
    fastmath=True,
)
def _cdl_dragonflydoji_nb(
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
    Optimized Dragonfly Doji pattern.

    Returns:
        1.0 → bullish dragonfly doji
       -1.0 → bearish dragonfly doji (rare, symmetric mode)
        0.0 → none
    """
    n = len(open_)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        o0 = open_[i]
        c0 = close[i]
        h0 = high[i]
        l0 = low[i]

        rng = h0 - l0
        if rng <= 0.0:
            continue

        # Body extremely small
        body = c0 - o0 if c0 > o0 else o0 - c0
        if body > min_body_factor * rng:
            continue

        # Dragonfly: open ≈ close ≈ high
        if not (o0 == h0 and c0 == h0):
            continue

        # Long lower shadow
        lower_shadow = h0 - l0
        if lower_shadow <= 0.0:
            continue

        if strict:
            # Upper shadow must be zero (already ensured by o0==h0 and c0==h0)
            # Lower shadow must dominate
            if lower_shadow < max_shadow_factor * rng:
                continue

        # Symmetric mode: direction by color
        if symmetric:
            if c0 > o0:
                out[i] = 1.0
            elif c0 < o0:
                out[i] = -1.0
            else:
                out[i] = 1.0  # perfect dragonfly
        else:
            out[i] = 1.0

    return out


def cdl_dragonflydoji(
    open_: np.ndarray | pl.Series,
    high: np.ndarray | pl.Series,
    low: np.ndarray | pl.Series,
    close: np.ndarray | pl.Series,
    offset: int = 0,
    fillna: float | None = None,
    use_talib: bool = True,
    strict: bool = False,
    symmetric: bool = False,
    min_body_factor: float = 0.1,
    max_shadow_factor: float = 0.5,
) -> np.ndarray:
    """
    Dragonfly Doji pattern with strict support.
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
        talib_out = talib.CDLDRAGONFLYDOJI(open_, high, low, close)
        talib_out = talib_out.astype(np.float64) / 100.0
        return _apply_offset_fillna(talib_out, offset, fillna)

    out = _cdl_dragonflydoji_nb(
        open_, high, low, close,
        min_body_factor, max_shadow_factor,
        strict, symmetric,
    )
    return _apply_offset_fillna(out, offset, fillna)


def cdl_dragonflydoji_polars(
    df: pl.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    offset: int = 0,
    fillna: float | None = None,
    strict: bool = False,
    symmetric: bool = False,
    min_body_factor: float = 0.1,
    max_shadow_factor: float = 0.5,
    output_col: str = "CDL_DRAGONFLYDOJI",
) -> pl.DataFrame:
    out = cdl_dragonflydoji(
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
