# System Architecture

## Overview

The Entry‑Exit Transformer Trading System is a modular Python framework that combines classical technical analysis
(order blocks, indicators) with a Transformer neural network to predict trade entry and exit points.
It is designed for iterative experimentation: from data preparation and label generation through supervised
training to semi‑supervised self‑training and deployment with DSL‑based filters.

The system is built around the following core components:

- **Data layer** – feature/label files in Parquet format, sliding‑window dataset (`TradingDataset`).
- **Feature engineering** – ATR calculation, order‑block distances, label generation from a mechanical strategy.
- **Model** – `EntryExitTransformer`, a dual‑encoder architecture that consumes time‑series data and order‑block information.
- **Loss** – `dual_loss`, which jointly optimises action classification, outcome prediction, and pattern detection.
- **Training** – supervised training with validation, metrics, early stopping, and self‑training on unlabeled data.
- **DSL** – a declarative language for expressing trading rules; can be used as an input feature or a post‑prediction filter.
- **Contracts** – strict validation of every batch before it reaches the model.

## Data Flow

1. **Raw data** → processed into Parquet files containing price data, indicators, signals, TP/SL levels, and optionally pattern labels.
2. **Order blocks** are loaded from a separate Parquet file.
3. **Features** are stacked into a 2D array, and `TradingDataset` creates sliding windows.
4. **DataLoader** yields batches of 11 elements, including action/outcome targets, pattern targets, and stable bar indices.
5. **Validation contracts** check each batch for shape, dtype, and value constraints.
6. **Model** receives the batch and outputs action, outcome, and pattern logits.
7. **Loss** is computed, gradients are back‑propagated, and metrics are logged.
8. **Self‑training** (optional) uses a separate unlabeled pool to generate pseudo‑labels, which are merged back into the labeled set.

## Component Interaction

features.parquet labels.parquet order_blocks.parquet
| | |
v v v
load_features load_labels load_order_blocks
| | |
+-------> merge_features_labels <+
|
v
data (2D array)
|
v
TradingDataset
|
v
DataLoader (with collate_ob)
|
v
validate_batch (optional)
|
v
EntryExitTransformer
|
v
dual_loss
|
v
Metrics & Logging

For self‑training, an additional unlabeled features file is used, and pseudo‑labels are written back to the original labels file.

## Key Design Decisions

- **11‑element batches** to support pattern targets and stable bar indices for safe pseudo‑label alignment.
- **Class weights** are computed from the training labels and passed to the loss to handle imbalance.
- **Validation split** can be a fraction of the training data or a separate file; if `val_split <= 0`, validation is skipped.
- **Checkpointing** saves the best model based on validation loss, and early stopping avoids overfitting.
- **DSL** is parsed at rule‑definition time and evaluated per bar via a context that provides indicator values.

## Next Steps

Read the detailed documentation for each component:

- [Overview](README.md)
- [Architecture](architecture.md)
- [Model](model.md)
- [Contract](contract.md)
- [Data Preparation](data.md)
- [Training & Self‑training](training.md)