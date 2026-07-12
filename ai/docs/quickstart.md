# Data Preparation

This document describes the required format and content of the three Parquet files that serve as input to the training pipeline.

## File Overview

| File                  | Purpose                                                     |
|-----------------------|-------------------------------------------------------------|
| `features.parquet`    | Time‑series data: prices, indicators, signals, TP/SL        |
| `labels.parquet`      | Target labels for supervised learning                       |
| `order_blocks.parquet`| Supply/demand zones used as context and for label generation|

All files must have the same number of rows, aligned bar‑by‑bar (or joined via a `bar_index` column).

---

## 1. features.parquet

Contains the input features for every bar. Columns can be grouped as follows:

### Price Columns (price_cols)

Typically OHLCV (open, high, low, close, volume). Must be positive floats.

Example:
open, high, low, close, volume
100.5, 101.2, 99.8, 100.9, 15000

text

### Indicator Columns (ind_cols) – optional

Technical indicators computed on the price data, e.g. RSI, MACD, ATR. Can be empty.

Example:
rsi, macd, macd_signal
42.3, 0.12, 0.08

text

### Signal Columns (sig_cols)

Derived features that the model uses as additional context. Common examples:

- Distances to nearest supply/demand zones (normalised by ATR)
- DSL rule evaluation results (binary 0/1)
- Custom pattern flags

Example:
dist_supply, dist_demand, dsl_oversold
1.23, 0.45, 1.0

text

### TP/SL Columns (tp_sl_cols)

Exactly two columns containing the **absolute price levels** for take‑profit and stop‑loss that would be used if an entry were to occur at this bar.

Example:
tp, sl
102.5, 98.0

text

### Pattern Columns (pattern_cols) – optional

Multi‑label indicators for candlestick patterns or other events. Values should be `0.0` or `1.0`.

Example:
doji, engulfing, pinbar
0.0, 1.0, 0.0

text

### Bar Index Column (bar_index) – optional but recommended

A stable integer identifier for each bar (e.g. a UNIX timestamp or sequential index). **Essential for safe pseudo‑label alignment during self‑training.** If not provided, positional indices are used.

---

## 2. labels.parquet

Contains the supervised learning targets.

### Required Columns

| Column    | Dtype | Description                                      | Valid Values                                                   |
|-----------|-------|--------------------------------------------------|----------------------------------------------------------------|
| `action`  | int   | Action label                                     | `-100` (ignore), `0` (hold), `1` (entry), `2` (exit)           |
| `outcome` | float | Outcome label (mode‑dependent)                   | `0.0`/`1.0` for binary; `2.0` for ignore; `NaN` for regression |

If pattern labels are provided separately, they should be in `features.parquet` or merged into this file.

---

## 3. order_blocks.parquet

Stores the supply and demand zones detected by your order‑block detection algorithm.

### Required Columns

| Column           | Type     | Description                                          |
|------------------|----------|------------------------------------------------------|
| `id`             | int      | Unique identifier                                    |
| `block_type`     | str      | `"supply"` or `"demand"`                             |
| `start`          | datetime | When the block started forming                       |
| `break_`         | datetime | When the block was broken                            |
| `retest`         | datetime | When the block was retested                          |
| `zone_low`       | float    | Lower boundary of the zone                           |
| `zone_high`      | float    | Upper boundary of the zone                           |
| `start_idx`      | int      | Index of the first bar in the features DataFrame     |
| `end_idx`        | int      | Index of the last bar                                |

### Optional Columns

| Column           | Type     | Description                              |
|------------------|----------|------------------------------------------|
| `strength`       | float    | Block strength (default 0.0)             |
| `structure_label`| str      | `"valid"`, `"broken"`, `"weak"`, or null |
| `trend_direction`| str      | `"up"`, `"down"`, or null                |

---

## Feature Engineering Utilities

The package provides ready‑to‑use functions for feature engineering:

### `compute_atr(df, period=14)`

Returns a NumPy array of ATR values, clipped to a minimum of 1e‑6.

### `compute_ob_distances(df, order_blocks, atr_series, close_col='close')`

Returns three arrays of ATR‑normalised distances:

- `nearest_supply`
- `nearest_demand`
- `strongest_dist` (distance to the strongest block)

### `generate_labels_from_strategy(df, order_blocks, min_rr, ...)`

Simulates a mechanical trading strategy and returns `(action, outcome)` arrays. Ideal for creating initial labels before model training.

---

## Example Workflow

```python
import polars as pl
from trading.io import load_features_parquet, save_labels_parquet, load_order_blocks_parquet
from trading.features import compute_atr, compute_ob_distances, generate_labels_from_strategy

# Load data
df = load_features_parquet("raw_features.parquet")
obs = load_order_blocks_parquet("detected_obs.parquet")

# Add OB distance features
atr = compute_atr(df)
ds, dd, ds2 = compute_ob_distances(df, obs, atr)
df = df.with_columns([
    pl.Series("dist_supply", ds),
    pl.Series("dist_demand", dd),
    pl.Series("dist_strong", ds2),
])

# Generate initial labels
action, outcome = generate_labels_from_strategy(df, obs, min_rr=1/3)

# Save
lbl_df = pl.DataFrame({"action": action, "outcome": outcome})
save_labels_parquet(lbl_df, "labels.parquet")
df.write_parquet("features.parquet")
```

After this, you can pass the three files to quick_train or the training pipeline.

## Next Steps

Read the detailed documentation for each component:

- [Overview](README.md)
- [Architecture](architecture.md)
- [Model](model.md)
- [Contract](contract.md)
- [Data Preparation](data.md)
- [Training & Self‑training](training.md)
