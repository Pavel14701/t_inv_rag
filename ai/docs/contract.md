# Batch Validation Contracts

The `contracts` module provides **runtime validation** of every batch before it reaches the model.
These checks catch common data errors early—missing normalisation, NaN values, incorrect shapes,
label outliers, and invalid order block definitions.

## Why Contracts?

In machine learning pipelines, data bugs often manifest as silent degradation of performance or
confusing error messages deep inside the model. Contracts turn those silent failures into **loud,
descriptive errors** before training starts.

The validation functions are designed to be:

- **Fast** – each check adds negligible overhead (< 0.1 ms per batch).
- **Comprehensive** – they cover tensor properties, label ranges, and domain‑specific constraints.
- **Optional but recommended** – they are not required for production inference, but they are
  invaluable during development and debugging.

## Usage

Call `validate_batch` at the beginning of the training loop, right after loading a batch from the
DataLoader:

```python
from trading.contracts import validate_batch

for batch in train_loader:
    validate_batch(
        batch,
        expected_batch_size=batch_size,
        seq_len=seq_len,
        n_price_feats=len(price_cols),
        n_ind_feats=len(ind_cols),
        n_sig_feats=len(sig_cols),
        outcome_mode=outcome_mode,
    )
    # ... training step ...
```

If any check fails, a ValueError or TypeError is raised with a descriptive message.

### Checks Performed

1. Batch structure (validate_batch)
The batch must contain exactly 10 elements:
prices, indicators, signals, tp, sl, order_blocks, action_targets, outcome_targets, pattern_targets, start_indices.
Each tensor must have the correct number of dimensions and consistent batch/sequence dimensions.
The sequence length (T) must equal seq_len.

2. Tensor basics (_check_tensor)
Applied to every tensor:
Must be a torch.Tensor.
Must have the expected number of dimensions.
dtype must be float32 or float64 (with a warning for float64). float16 triggers a warning
about using mixed precision.
Must contain no NaN or Inf values.

3. Price tensor (validate_prices)
Shape must be (B, T, n_price_feats).
All values should be positive. Non‑positive values trigger a warning about missing normalisation
(e.g., z‑score or ATR scaling).

4. Indicators and signals (validate_indicators, validate_signals)
If n_ind_feats or n_sig_feats is greater than 0, the last dimension must match exactly.
If the declared number of features is 0 but the tensor has non‑zero size, a warning is printed
(the model will ignore the extra features).

5. TP/SL levels (validate_tp_sl)
Both tp and sl must have shape (B, T, 1).
Their shapes must match each other.
All values must be strictly positive (absolute prices).

6. Order blocks (validate_order_blocks)
Must be a list of lists (List[List[OrderBlock]]).
The outer list must have length B (batch size).
Each inner list must contain only OrderBlock instances.
For every order block:
start_idx and end_idx must be non‑negative.
end_idx must be less than seq_len.
zone_low must be strictly less than zone_high.
strength must be ≥ 0.

7. Action targets (validate_action_targets)
Shape (B, T).
All values must be in the allowed set: {-100, 0, 1, 2}.

8. Outcome targets (validate_outcome_targets)
Shape (B, T).
In binary or multiclass mode, finite values must be in {0.0, 1.0, 2.0}. NaN is allowed
(used as ignore in regression).

In regression mode, a warning is issued if absolute values exceed 1e6.

9. Pattern targets (informal check)
dtype should be float32. If not, a warning is printed.

10. Start indices (start_indices)
Shape must be (B,).
Provided for completeness; not range‑checked.
Individual Validator Functions
All validators are importable and can be used independently:

```python
from trading.contracts import (
    validate_prices,
    validate_indicators,
    validate_signals,
    validate_tp_sl,
    validate_order_blocks,
    validate_action_targets,
    validate_outcome_targets,
)
```

Each follows the pattern:

```python
def validate_<something>(tensor, expected_param, ...):
    """Check <something> and raise on error."""
```

### Customisation

The checks are deliberately conservative. If you need to relax or strengthen them:
Modify the allowed_dtypes tuple in `_check_tensor` (e.g., allow float16).
Adjust the warning thresholds (e.g., the 1e6 limit in validate_outcome_targets).
Override validate_batch with your own wrapper that calls a subset of checks.
The module is pure Python and depends only on torch and your OrderBlock definition.

### Performance Notes

Validating a full batch of 16 sequences of length 128 takes < 1 ms on a modern CPU.
The checks are purely CPU‑based and do not require GPU transfer.
For production inference where throughput is critical, you can disable validation with a simple
conditional or by removing the validate_batch call.

### Error Messages

All error messages include the name of the tensor or field that failed, making debugging fast:

```text
ValueError: prices: contains NaN or Inf
ValueError: tp and sl shapes mismatch: torch.Size([16, 128, 1]) vs torch.Size([16, 127, 1])
ValueError: action_targets contains invalid values: [3, 4]. Allowed: {-100, 0, 1, 2}
ValueError: OrderBlock 5 at order_blocks[2][0]: zone_low must be < zone_high
```

#### This lets you pinpoint the problematic batch or feature immediately.

Integration Example
python
import torch
from trading.contracts import validate_batch

## Assume train_loader yields 10‑element batches

```python
for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        try:
            validate_batch(
                batch,
                expected_batch_size=16,
                seq_len=128,
                n_price_feats=5,
                n_ind_feats=0,
                n_sig_feats=3,
                outcome_mode='binary',
            )
        except (ValueError, TypeError) as e:
            print(f"Batch {batch_idx} failed validation: {e}")
            continue  # skip this batch, or re-raise

        # ... training step ...
```

This pattern is especially useful when experimenting with new data sources or feature engineering.

### Summary

- 10‑element batches are validated for shape, dtype, and value consistency.
- Domain‑specific checks ensure order blocks, TP/SL, and labels are semantically valid.
- Development‑time safety net – catches errors early, before they corrupt model training.
- Zero overhead for production – can be disabled with a single line change.

## Next Steps

Read the detailed documentation for each component:

- [Overview](README.md)
- [Architecture](architecture.md)
- [Model](model.md)
- [Contract](contract.md)
- [Data Preparation](data.md)
- [Training & Self‑training](training.md)