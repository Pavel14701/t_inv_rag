# Training & Self‑Training

This document describes the supervised training loop, validation metrics, and the semi‑supervised self‑training procedure.

---

## Supervised Training – `train_one_round`

The function `train_one_round` runs a standard supervised training loop with extensive monitoring.

### Parameters

| Parameter                | Type                    | Default     | Description                                                   |
|--------------------------|-------------------------|-------------|---------------------------------------------------------------|
| `model`                  | `nn.Module`             | required    | The transformer model                                         |
| `train_loader`           | `DataLoader`            | required    | Training data                                                 |
| `val_loader`             | `DataLoader` or `None`  | required    | Validation data; if `None`, validation is skipped             |
| `epochs`                 | `int`                   | required    | Number of epochs                                              |
| `device`                 | `torch.device`          | required    | Torch device                                                  |
| `outcome_mode`           | `str`                   | `"binary"`  | One of `"binary"`, `"multiclass"`, `"regression"`             |
| `lambda_outcome`         | `float`                 | `0.3`       | Weight of outcome loss                                        |
| `lr`                     | `float`                 | `1e-4`      | Learning rate (AdamW)                                         |
| `lambda_pattern`         | `float`                 | `0.1`       | Weight of pattern loss (0 disables)                           |
| `class_weight`           | `Tensor` or `None`      | `None`      | Class weights for action cross‑entropy                        |
| `log_dir`                | `str` or `None`         | `None`      | TensorBoard log directory                                     |
| `save_best`              | `bool`                  | `True`      | Whether to save the best model                                |
| `best_model_path`        | `str` or `None`         | `None`      | Path for the best checkpoint                                  |
| `early_stopping_patience`| `int`                   | `0`         | Stop after this many epochs without improvement (0 = off)     |
| `close_idx`              | `int`                   | `3`         | Index of close price (for legacy API)                         |

### Internal Workflow

1. **Training epoch** (`_run_train_epoch`):  
   - Iterates over `train_loader`.  
   - Computes loss via `_model_forward_loss` → `dual_loss`.  
   - Back‑propagates and updates weights.

2. **Validation epoch** (`_run_val_epoch`), skipped if `val_loader is None`:  
   - Computes loss without gradient.  
   - Accumulates flattened action logits, action targets, and outcome targets.

3. **Metrics** (`_compute_val_metrics`):  
   - `compute_action_accuracy` – overall accuracy and per‑class (hold/entry/exit) accuracy.  
   - `compute_trade_metrics` – win rate and profit factor based on bars where predicted and true action are both entry.

4. **Logging** (`_log_epoch`):  
   - Prints a one‑line summary to the console.  
   - Writes scalars to TensorBoard if `log_dir` is provided.

5. **Scheduler step**: `ReduceLROnPlateau` adjusts the learning rate based on validation loss.

6. **Checkpointing & Early Stopping** (`_checkpoint_and_stop`):  
   - Saves the model state when validation loss improves.  
   - Stops training if no improvement for `early_stopping_patience` epochs.

### Example

```python
from trading.training import train_one_round

model = train_one_round(
    model,
    train_loader,
    val_loader,
    epochs=20,
    device=torch.device("cuda"),
    outcome_mode="binary",
    lambda_outcome=0.3,
    lr=1e-4,
    save_best=True,
    best_model_path="best_model.pt",
    early_stopping_patience=5,
    log_dir="./logs"
)
Self‑Training – self_training_loop
Self‑training uses a separate pool of unlabeled data to iteratively improve the model.

How It Works
Initial training on the labeled set (with optional validation split).

Pseudo‑label generation on the unlabeled pool:

The model makes predictions on unlabeled bars.

A pseudo‑label is created only if:

Action probability exceeds action_threshold.

For entry predictions, the trade passes a risk‑reward check (min_rr).

Outcome confidence exceeds outcome_threshold.

The stable bar index (from the bar_index column, or positional index) is used to match pseudo‑labels to rows in labels.parquet.

Label update: The generated pseudo‑labels are written back to labels.parquet.

Re‑training: The labeled loaders are rebuilt with the expanded labels, and the cycle repeats.


| Parameter                  | Type                | Default    | Description                                                                 |
|----------------------------|---------------------|------------|-----------------------------------------------------------------------------|
| `model`                    | `torch.nn.Module`   | required   | Initialised model                                                           |
| `features_path`            | `str`               | required   | Labeled features Parquet file                                               |
| `labels_path`              | `str`               | required   | Labels Parquet file (updated in‑place)                                      |
| `features_path_unlabeled`  | `str`               | required   | Unlabeled features Parquet file                                             |
| `order_blocks`             | `list[OrderBlock]`  | required   | All order blocks                                                            |
| `price_cols`               | `list[str]`         | required   | Names of price columns                                                      |
| `ind_cols`                 | `list[str]`         | required   | Names of indicator columns (can be empty)                                   |
| `sig_cols`                 | `list[str]`         | required   | Names of signal columns                                                     |
| `tp_sl_cols`               | `list[str]`         | required   | Names of TP/SL columns                                                      |
| `seq_len`                  | `int`               | required   | Sequence length (number of bars per window)                                 |
| `batch_size`               | `int`               | required   | Batch size                                                                  |
| `device`                   | `torch.device`      | required   | PyTorch device                                                              |
| `outcome_mode`             | `str`               | `"binary"` | Outcome prediction mode (`"binary"`, `"multiclass"`, `"regression"`)        |
| `lambda_outcome`           | `float`             | `0.3`      | Weight of outcome loss                                                      |
| `lr`                       | `float`             | `1e-4`     | Learning rate (AdamW)                                                       |
| `epochs_per_round`         | `int`               | `3`        | Training epochs per round                                                   |
| `num_rounds`               | `int`               | `3`        | Maximum number of self‑training rounds                                      |
| `action_threshold`         | `float`             | `0.9`      | Minimum probability for a predicted action to be accepted                   |
| `outcome_threshold`        | `float`             | `0.8`      | Minimum confidence for outcome to be accepted                               |
| `min_rr`                   | `float`             | `1/3`      | Minimum risk‑reward ratio for entry pseudo‑labels                           |
| `close_idx`                | `int`               | `3`        | Index of close price within price features                                  |
| `save_model_path`          | `str` or `None`     | `None`     | Base path for saving round checkpoints                                      |
| `val_split`                | `float`             | `0.2`      | Fraction of labeled data used for validation (`0` = no validation)          |
| `log_dir`                  | `str` or `None`     | `None`     | TensorBoard log directory                                                   |
| `early_stopping_patience`  | `int`               | `0`        | Stop training after this many epochs without improvement (`0` = disabled)   |

Safety Notes
Pseudo‑labels are only written for unlabeled bars (action == -100). Already labeled bars are never overwritten.

The stable bar index ensures that pseudo‑labels are mapped to the correct rows in labels.parquet even if the unlabeled file has a different number of rows or ordering.

The risk‑reward check prevents the model from labelling entries with unrealistic TP/SL ratios.

Example
```python
from trading.training import self_training_loop

model = self_training_loop(
    model,
    features_path="labeled_features.parquet",
    labels_path="labels.parquet",
    features_path_unlabeled="unlabeled_features.parquet",
    order_blocks=obs,
    price_cols=["open","high","low","close","volume"],
    ind_cols=[],
    sig_cols=["dist_supply","dist_demand"],
    tp_sl_cols=["tp","sl"],
    seq_len=128,
    batch_size=16,
    device=torch.device("cuda"),
    val_split=0.2,
    epochs_per_round=5,
    num_rounds=3,
    action_threshold=0.95,
    outcome_threshold=0.8,
    min_rr=1/3,
    save_model_path="st_model",
)
```

Metrics
Two metric functions are provided:

compute_action_accuracy(action_logits, action_targets, ignore_index=-100)
Returns a dictionary with keys overall, hold, entry, exit. Computed only on bars where the target is not the ignore index.

compute_trade_metrics(action_logits, action_targets, outcome_targets, ignore_index=2)
Returns win_rate, profit_factor, and num_trades based on bars where both predicted and true action are entry.

win_rate = fraction of such bars where outcome == 1.0.

profit_factor = wins / losses (only for binary outcomes).

These metrics are a rough monitoring tool, not a full backtest.

Class Weighting
When action labels are imbalanced (e.g., far more hold than entry), class weights can be computed and passed to train_one_round:

```python
from trading.training import _compute_class_weights
cw = _compute_class_weights(df["action"].to_numpy())
model = train_one_round(..., class_weight=cw)
The weights are normalised so that their mean is 1.
```

## Next Steps

Read the detailed documentation for each component:

- [Overview](README.md)
- [Architecture](architecture.md)
- [Model](model.md)
- [Contract](contract.md)
- [Data Preparation](data.md)
- [Training & Self‑training](training.md)
