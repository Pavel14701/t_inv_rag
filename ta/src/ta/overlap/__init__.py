# -*- coding: utf-8 -*-
from .alligator import alligator_ind, alligator_polars
from .alma import alma_ind, alma_polars
from .dema import dema_ind, dema_polars
from .ema import ema_ind, ema_polars
from .fwma import fwma_ind, fwma_polars
from .hilo import hilo_ind, hilo_polars
from .hl2 import hl2_ind, hl2_polars
from .hlc3 import hlc3_ind, hlc3_polars
from .hma import hma_ind, hma_polars
from .hwma import hwma_ind, hwma_polars
from .ichimoku import ichimoku_ind
from .jma import jma_ind, jma_polars
from .kama import kama_ind, kama_polars
from .linreg import linreg_ind, linreg_polars
from .mama import mama_ind, mama_polars

#from .mcgd import mcgd
#from .midpoint import midpoint
#from .midprice import midprice
#from .ohlc4 import ohlc4
#from .pivots import pivots
#from .pwma import pwma
from .rma import rma_ind, rma_polars

#from .sinwma import sinwma
from .sma import sma_ind, sma_polars
from .smma import smma_ind, smma_polars

#from .ssf import ssf
#from .ssf3 import ssf3
#from .supertrend import supertrend
#from .swma import swma
#from .t3 import t3
#from .tema import tema
#from .trima import trima
#from .vidya import vidya
#from .wcp import wcp
from .wma import wma_ind, wma_polars

#from .zlma import zlma


__all__ = [
    "alligator_ind", "alligator_polars",
    "alma_ind", "alma_polars",
    "dema_ind", "dema_polars",
    "ema_ind", "ema_polars",
    "fwma_ind", "fwma_polars",
    "hilo_ind", "hilo_polars",
    "hl2_ind", "hl2_polars",
    "hlc3_ind", "hlc3_polars",
    "hma_ind", "hma_polars",
    "hwma_ind", "hwma_polars",
    "ichimoku_ind",
    "jma_ind", "jma_polars",
    "kama_ind", "kama_polars",
    "linreg_ind", "linreg_polars",
    "mama_ind", "mama_polars",
#    "mcgd",
#    "midpoint",
#    "midprice",
#    "ohlc4",
#    "pivots",
#    "pwma",
    "rma_ind", "rma_polars",
#    "sinwma",
    "sma_ind", "sma_polars",
    "smma_ind", "smma_polars",
#    "ssf",
#    "ssf3",
#    "supertrend",
#    "swma",
#    "t3",
#    "tema",
#    "trima",
#    "vidya",
#    "wcp",
    "wma_ind", "wma_polars", 
#    "zlma",
]
