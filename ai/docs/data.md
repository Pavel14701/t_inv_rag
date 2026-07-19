# Data Preparation

This document describes how to prepare the three Parquet files required by the Entry‑Exit Transformer trading system:

| File                   | Purpose                                                                                   |
|------------------------|-------------------------------------------------------------------------------------------|
| `features.parquet`     | Time‑series data: prices, indicators, signals, TP/SL levels, and optional pattern labels. |
| `labels.parquet`       | Target labels for supervised learning (action and outcome).                               |
| `order_blocks.parquet` | Supply/demand zones used as context and for label generation.                             |

All files must be aligned by row – each row corresponds to one bar (candlestick). Optionally, a `bar_index` column can be used to join files that may not be perfectly aligned.

---

## 1. features.parquet

This file contains all input features. Columns are grouped by their role:

### 1.1 Price Columns (`price_cols`)

**Required.** At minimum, OHLC (open, high, low, close). Volume is recommended.

| Column   | Dtype            | Description              |
|----------|------------------|--------------------------|
| `open`   | float            | Opening price of the bar |
| `high`   | float            | Highest price            |
| `low`    | float            | Lowest price             |
| `close`  | float            | Closing price            |
| `volume` | float (optional) | Traded volume            |

**Important:** All price values must be **positive** and **non‑zero**. The validation contracts will warn if non‑positive values are detected (they suggest missing normalisation).

### 1.2 Indicator Columns (`ind_cols`) – optional

Technical indicators pre‑computed from price data.

| Column        | Dtype | Description         |
|---------------|-------|---------------------|
| `rsi`         | float | RSI value (0–100)   |
| `macd`        | float | MACD line           |
| `macd_signal` | float | MACD signal line    |
| `atr`         | float | Average True Range  |
| …             | float | Any other indicator |

If no indicators are used, pass an empty list to `ind_cols`. The dataset will handle this gracefully.

### 1.3 Signal Columns (`sig_cols`)

Derived features that the model uses as additional context. These are typically computed from price, indicators, and order blocks.

| Column        | Dtype           | Description                                    |
|---------------|-----------------|------------------------------------------------|
| `dist_supply` | float           | ATR‑normalised distance to nearest supply zone |
| `dist_demand` | float           | ATR‑normalised distance to nearest demand zone |
| `dist_strong` | float           | Distance to the strongest order block          |
| `dsl_rule_1`  | float (0.0/1.0) | Binary signal from a DSL rule                  |
| …             | float           | Any custom signal                              |

**At minimum**, you need at least one signal column (even if it’s a dummy zero column). The model expects `n_sig_feats` to match the number of columns listed in `sig_cols`.

### 1.4 TP/SL Columns (`tp_sl_cols`)

**Required.** Exactly two columns containing the **absolute price levels** for take‑profit and stop‑loss that would be used if an entry were triggered at this bar.

| Column | Dtype | Description                     |
|--------|-------|---------------------------------|
| `tp`   | float | Take‑profit price (must be > 0) |
| `sl`   | float | Stop‑loss price (must be > 0)   |

**Validation:** Both values must be strictly positive. The validation contract will reject batches with non‑positive TP/SL.

### 1.5 Pattern Columns (`pattern_cols`) – optional

Multi‑label binary indicators for candlestick patterns or other events.

| Column      | Dtype | Description                                     |
|-------------|-------|-------------------------------------------------|
| `doji`      | float | 1.0 if a doji pattern is present, 0.0 otherwise |
| `engulfing` | float | 1.0 if engulfing pattern                        |
| `pinbar`    | float | 1.0 if pin bar                                  |
| …           | float | 0.0 or 1.0                                      |

The pattern head of the model is **only trained** when `pattern_cols` is provided.

### 1.6 Bar Index Column (`bar_index`) – optional but recommended

A stable integer identifier for each bar. Essential for safe pseudo‑label alignment during self‑training.

| Column      | Dtype | Description                                                        |
|-------------|-------|--------------------------------------------------------------------|
| `bar_index` | int64 | Unique, sequential identifier (e.g., UNIX timestamp or row number) |

If absent, the system falls back to positional indices.

---

## 2. labels.parquet

Contains the supervised learning targets.

### Required Columns

| Column    | Dtype | Description   | Valid Values                                                       |
|-----------|-------|---------------|--------------------------------------------------------------------|
| `action`  | int   | Action label  | `-100` (ignore), `0` (hold), `1` (entry), `2` (exit)               |
| `outcome` | float | Outcome label | `0.0` (loss), `1.0` (win), `2.0` (ignore), `NaN` (regression mode) |

**Important:**

- Bars where no trading event occurs should be marked with `action = -100` (ignored by the loss).
- In binary mode, `outcome = 2.0` is the ignore value (e.g., for bars that are not entry bars).
- In regression mode, `outcome = NaN` is the ignore value.

### Example

| action | outcome |
|--------|---------|
| -100   | 2.0     |
| 1      | 1.0     |
| 2      | 2.0     |
| -100   | 2.0     |

---

## 3. order_blocks.parquet

Stores the supply and demand zones detected by your order‑block detection algorithm.

### Required Columns

| Column       | Dtype    | Description                                      |
|--------------|----------|--------------------------------------------------|
| `id`         | int      | Unique identifier for the block                  |
| `block_type` | str      | `"supply"` or `"demand"`                         |
| `start`      | datetime | When the block started forming                   |
| `break_`     | datetime | When the block was broken                        |
| `retest`     | datetime | When the block was retested                      |
| `zone_low`   | float    | Lower boundary of the price zone                 |
| `zone_high`  | float    | Upper boundary of the price zone                 |
| `start_idx`  | int      | Index of the first bar in the features DataFrame |
| `end_idx`    | int      | Index of the last bar                            |

### Optional Columns

| Column            | Dtype       | Default | Description                                |
|-------------------|-------------|---------|--------------------------------------------|
| `strength`        | float       | `0.0`   | Block strength (higher = more significant) |
| `structure_label` | str or None | `None`  | `"valid"`, `"broken"`, `"weak"`            |
| `trend_direction` | str or None | `None`  |        `"up"`, `"down"`                    |

**Validation:** The validation contract checks that `zone_low < zone_high`, `strength >= 0`, and that `start_idx`/`end_idx` are valid.

---

## Feature Engineering Pipeline

The package provides ready‑to‑use functions to compute the required features.

### Step 1: Load raw data

```python
import polars as pl
from trading.io import load_features_parquet, load_order_blocks_parquet

df = load_features_parquet("raw_data.parquet")  # must contain OHLCV
obs = load_order_blocks_parquet("detected_obs.parquet")
```

### Step 2: Compute ATR

```python
from trading.features import compute_atr

atr = compute_atr(df, period=14)  # returns float32 array
```

### Step 3: Compute order‑block distances

```python
from trading.features import compute_ob_distances

dist_supply, dist_demand, dist_strong = compute_ob_distances(df, obs, atr)
df = df.with_columns([
    pl.Series("dist_supply", dist_supply),
    pl.Series("dist_demand", dist_demand),
    pl.Series("dist_strong", dist_strong),
])
```

### Step 4: Add indicators (example)

```python
# Example: add RSI (pseudo‑code; use your own indicator library)
df = df.with_columns([
    pl.Series("rsi", compute_rsi(df["close"], period=14)),
])
```

### Step 5: (Optional) Add DSL signals

```python
from trading.dsl import evaluate_dsl, DataFrameContext

rule = "rsi < 30 and close > 100"
signal = [float(evaluate_dsl(rule, DataFrameContext(df, i))) for i in range(len(df))]
df = df.with_columns(pl.Series("oversold_signal", signal))
```

### Step 6: Save features

```python
df.write_parquet("features.parquet")
```

### Step 7: Generate initial labels

```python
from trading.features import generate_labels_from_strategy

action, outcome = generate_labels_from_strategy(
    df, obs,
    min_rr=1/3,
    use_r_multiple=False,
    use_structure_filter=False,
)

lbl_df = pl.DataFrame({"action": action, "outcome": outcome})
lbl_df.write_parquet("labels.parquet")
```

After these steps, you have all three files ready for training.

Data Alignment
features.parquet and labels.parquet must have the same number of rows, aligned bar‑by‑bar.

order_blocks.parquet uses start_idx/end_idx to reference bars in features.parquet – ensure these indices are correct.

If using bar_index for joining, the column must exist in both files and contain matching values.

### Quick‑Start Example

```python
# 1. Load raw OHLCV data
df = pl.read_parquet("raw_ohlcv.parquet")

# 2. Load order blocks (pre‑detected)
obs = load_order_blocks_parquet("order_blocks.parquet")

# 3. Compute features
atr = compute_atr(df)
ds, dd, dstrong = compute_ob_distances(df, obs, atr)
df = df.with_columns([
    pl.Series("dist_supply", ds),
    pl.Series("dist_demand", dd),
])

# 4. Add TP/SL levels (example: fixed 2% stop, 3% take profit)
df = df.with_columns([
    (df["close"] * 1.03).alias("tp"),
    (df["close"] * 0.98).alias("sl"),
])

# 5. Save features
df.write_parquet("features.parquet")

# 6. Generate labels from strategy
action, outcome = generate_labels_from_strategy(df, obs, min_rr=1/3)
pl.DataFrame({"action": action, "outcome": outcome}).write_parquet("labels.parquet")

# 7. Train model
from trading.quickstart import quick_train
model = quick_train(
    features="features.parquet",
    labels="labels.parquet",
    order_blocks="order_blocks.parquet",
    price_cols=["open","high","low","close"],
    sig_cols=["dist_supply","dist_demand"],
    tp_sl_cols=["tp","sl"],
    epochs=20,
    batch_size=16,
)
```

This gives you a fully trained model in a few lines of code.

## Next Steps

Read the detailed documentation for each component:

- [Overview](README.md)
- [Architecture](architecture.md)
- [Model](model.md)
- [Contract](contract.md)
- [Data Preparation](data.md)
- [Training & Self‑training](training.md)
