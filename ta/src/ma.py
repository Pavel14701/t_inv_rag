from typing import Callable

import numpy as np
import polars as pl

from .overlap import (
    dema_ind,
    ema_ind,
    fwma_ind,
    hma_ind,
    kama_ind,
    linreg_ind,
    midpoint_ind,
    pwma_ind,
    rma_ind,
    sinwma_ind,
    sma_ind,
    ssf_ind,
    swma_ind,
    t3_ind,
    tema_ind,
    trima_ind,
    vidya_ind,
    wma_ind,
)

_MA_MAP: dict[str, Callable[..., np.ndarray]] = {
    "sma": sma_ind,
    "ema": ema_ind,
    "wma": wma_ind,
    "dema": dema_ind,
    "tema": tema_ind,
    "trima": trima_ind,
    "kama": kama_ind,
    "hma": hma_ind,
    "fwma": fwma_ind,
    "pwma": pwma_ind,
    "sinwma": sinwma_ind,
    "swma": swma_ind,
    "rma": rma_ind,
    "vidya": vidya_ind,
    "linreg": linreg_ind,
    "midpoint": midpoint_ind,
    "ssf": ssf_ind,
    "t3": t3_ind,
}


def ma_mode(
    mamode: str | None = None,
    source: np.ndarray | pl.Series | None = None,
    **kwargs,
) -> list[str] | np.ndarray:
    if mamode is None and source is None:
        return list(_MA_MAP.keys())
    if source is None:
        raise ValueError("source must be provided when name is given")
    if mamode:
        mamode = mamode.lower()
        func = _MA_MAP.get(mamode)
    else:
        func = ema_ind
    if func is None:
        func = ema_ind
    if isinstance(source, pl.Series):
        source = source.to_numpy()
    return func(source, **kwargs)