# EntryExitTransformer – Model Architecture

## Overview

`EntryExitTransformer` is a dual‑encoder Transformer model that processes a fixed‑length window of market data together with a set of order blocks (supply/demand zones) and produces per‑bar predictions for:

- **action** – 3‑class classification: hold (0), entry (1), exit (2)
- **outcome** – binary, multiclass, or regression (e.g., R‑multiple)
- **pattern** – optional multi‑label classification (trained only when pattern labels are available)

The model is designed to be **invariant to the number of order blocks** in the window and to **scale to sequences of up to 1024 bars**.

## Inputs

| Tensor            | Shape                       | Description                                    |
|-------------------|-----------------------------|------------------------------------------------|
| `prices`          | `(B, T, n_price_feats)`     | OHLCV or similar price data                    |
| `indicators`      | `(B, T, n_ind_feats)`       | Technical indicators (can be empty)            |
| `signals`         | `(B, T, n_sig_feats)`       | Derived signals (e.g., OB distances, DSL rules)|
| `tp_levels`       | `(B, T, 1)`                 | Take‑profit absolute price for each bar        |
| `sl_levels`       | `(B, T, 1)`                 | Stop‑loss absolute price                       |
| `order_blocks`    | `List[List[OrderBlock]]`    | Per‑sample list of OB objects in the window    |

All input tensors must be `float32`, have no NaN or Inf, and have consistent batch and sequence lengths.

## Architecture

### 1. Time Encoder

- **Input projection**: Linear layer merges all time‑series features (prices + indicators + signals + TP/SL) into a hidden vector.
- **Positional encoding**: Sinusoidal encoding added to the projected sequence.
- **Transformer Encoder**: Stack of `num_layers` standard Transformer encoder layers with GELU activation.
- **Output**: `(B, T, hidden_size)`

### 2. Order‑Block Encoder

Each OrderBlock is encoded into a combination of:

- **Numeric features** (5‑dim): normalised start/end indices, zone boundaries (scaled by global ATR), and strength.
- **Categorical embeddings**:
  - Block type: `supply` (0) or `demand` (1)
  - Structure label: `valid`, `broken`, `weak`, `None`
  - Trend direction: `up`, `down`, `None`

The numeric and categorical parts are merged via a linear layer into a vector of size `hidden_size`.

- **Sequence construction**: OB vectors are padded to a maximum length, and a learnable CLS token is prepended.
- **Positional encoding + Transformer Encoder**: same as the time encoder but with a separate set of layers.
- **Global OB representation**: The output at the CLS token position `(B, hidden_size)` is taken as the summary of all order blocks.

### 3. Feature Combination

The global OB vector is expanded to match the time dimension and concatenated with the time encoder output:
combined = concat(time_out, ob_global_expanded) # (B, T, 2 * hidden_size)s

### 4. Heads

Three independent MLP heads (each with GELU, dropout) operate on the `combined` tensor:

- **Action head**: `Linear → GELU → Dropout → Linear(→ 3)`
- **Outcome head**: structure depends on `outcome_mode`:
  - `binary` / `regression`: output size 1
  - `multiclass`: output size `n_outcome_classes`
- **Pattern head**: output size `n_patterns` (multi‑label logits)

All heads produce per‑bar predictions.

## Configuration Parameters

| Parameter          | Default | Description                                         |
|--------------------|---------|-----------------------------------------------------|
| `n_price_feats`    | required| Number of price features (e.g., 5 for OHLCV)        |
| `n_ind_feats`      | 3       | Number of indicator features (0 if none)            |
| `n_sig_feats`      | 2       | Number of signal features                           |
| `n_tp_sl_feats`    | 2       | Usually 2: tp and sl                                |
| `hidden_size`      | 128     | Transformer hidden dimension                        |
| `num_layers`       | 4       | Number of encoder layers                            |
| `num_heads`        | 8       | Attention heads                                     |
| `dropout`          | 0.1     | Dropout rate                                        |
| `max_seq_len`      | 1024    | Max time steps (positional encoding size)           |
| `max_ob_seq_len`   | 256     | Max number of order blocks per window               |
| `n_action_classes` | 3       | Always 3 (hold/entry/exit)                          |
| `outcome_mode`     | 'binary'| 'binary', 'multiclass', or 'regression'             |
| `n_outcome_classes`| 2       | Only used for multiclass                            |
| `n_patterns`       | 10      | Dimension of pattern head (0 disables)              |
| `ob_embedding_dim` | 32      | Dimension of categorical OB embeddings              |
| `atr_global`       | 1.0     | Global ATR value for normalising OB zones           |

## Forward Pass

```python
action_logits, outcome_logits, pattern_logits = model(
    prices, indicators, signals, tp_levels, sl_levels, order_blocks
)
```

action_logits: (B, T, 3)

outcome_logits: (B, T, 1) (or n_outcome_classes)

pattern_logits: (B, T, n_patterns)

Notes:
The pattern head is not trained unless pattern_targets are provided.

All layers use batch_first=True for compatibility with standard DataLoader outputs.

The model is fully deterministic; no random operations occur during inference.

## Next Steps

Read the detailed documentation for each component:

- [Overview](README.md)
- [Architecture](architecture.md)
- [Model](model.md)
- [Contract](contract.md)
- [Data Preparation](data.md)
- [Training & Self‑training](training.md)
