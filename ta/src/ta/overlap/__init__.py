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
from .mcgd import mcgd_ind, mcgd_polars
from .midpoint import midpoint_ind, midpoint_polars
from .midprice import midprice_ind, midprice_polars
from .ohlc4 import ohlc4_ind, ohlc4_polars
from .pivots import pivots_ind
from .pwma import pwma_ind, pwma_polars
from .rma import rma_ind, rma_polars
from .sinwma import sinwma_ind, sinwma_polars
from .sma import sma_ind, sma_polars
from .smma import smma_ind, smma_polars
from .ssf import ssf_ind, ssf_polars
from .ssf3 import ssf3_ind, ssf3_polars
from .supertrend import supertrend_ind, supertrend_polars
from .swma import swma_ind, swma_polars
from .t3 import t3_ind, t3_polars
from .tema import tema_ind, tema_polars
from .trima import trima_ind, trima_polars
from .vidya import vidya_ind, vidya_polars
from .wcp import wcp_ind, wcp_polars
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
    "mcgd_ind", "mcgd_polars",
    "midpoint_ind", "midpoint_polars",
    "midprice_ind", "midprice_polars",
    "ohlc4_ind", "ohlc4_polars",
    "pivots_ind",
    "pwma_ind", "pwma_polars",
    "rma_ind", "rma_polars",
    "sinwma_ind", "sinwma_polars",
    "sma_ind", "sma_polars",
    "smma_ind", "smma_polars",
    "ssf_ind", "ssf_polars",
    "ssf3_ind", "ssf3_polars",
    "supertrend_ind", "supertrend_polars",
    "swma_ind", "swma_polars",
    "t3_ind", "t3_polars",
    "tema_ind", "tema_polars",
    "trima_ind", "trima_polars",
    "vidya_ind", "vidya_polars",
    "wcp_ind", "wcp_polars",
    "wma_ind", "wma_polars", 
#    "zlma",
]
