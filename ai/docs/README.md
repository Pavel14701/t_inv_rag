# Entry-Exit Transformer Trading System

A modular, extensible framework for **order-block-aware trade signal prediction** using a Transformer neural network.
The system supports supervised learning, semi‑supervised self‑training, a built-in DSL for human‑readable trading rules, and strict data validation contracts.

## Key Features

- **Multi‑head Transformer** that predicts *action* (hold/entry/exit), *outcome* (win/loss or R‑multiple), and *patterns* from a window of price data and order blocks.
- **Self‑training** with a separate unlabeled data pool – pseudo‑labels are generated only for high‑confidence predictions and safely merged back into the labeled set.
- **DSL (Domain‑Specific Language)** for declarative trading rules – rules can be used as input features or as post‑prediction filters.
- **Rich validation contracts** that check tensor shapes, dtypes, finite values, and order‑block integrity before training.
- **Metrics & logging** – per‑epoch action accuracy, win rate, profit factor, and TensorBoard integration.
- **Production‑ready utilities** – class weighting, early stopping, best‑model checkpointing, and a one‑line `quick_train` entry point.

## Repository Structure

trading/
├── datatypes.py # OrderBlock dataclass
├── features.py # ATR, OB distance features, label generation
├── io.py # Parquet read/write, merge helpers
├── dataset.py # TradingDataset (sliding windows), collate_ob
├── losses.py # dual_loss (action + outcome + pattern)
├── transformer.py # EntryExitTransformer model
├── training.py # train_one_round, self_training_loop, loaders
├── metrics.py # compute_action_accuracy, compute_trade_metrics
├── contracts.py # validate_batch and internal checks
├── quickstart.py # quick_train – one‑function training
├── dsl/ # Domain‑Specific Language
│ ├── grammar.lark # Lark grammar
│ ├── ast.py # AST node classes
│ ├── parser.py # Parser (LALR)
│ ├── interpreter.py # Interpreter (Visitor pattern)
│ ├── context.py # Abstract Context interface
│ └── init.py # evaluate_dsl
└── init.py # Package root, exports, and documentation
docs/
├── README.md # This file
├── architecture.md # System architecture
├── model.md # Model details
├── data.md # Data preparation and formats
├── training.md # Training loop, metrics, self‑training
├── dsl.md # DSL language reference
├── contracts.md # Batch validation contracts
└── quickstart.md # Quick‑start guide

## Quick Start

After preparing your Parquet files (see `docs/data.md`), train a model in one line:

```python
from trading.quickstart import quick_train

model = quick_train(
    features="features.parquet",
    labels="labels.parquet",
    order_blocks="order_blocks.parquet",
    price_cols=["open","high","low","close","volume"],
    sig_cols=["dist_supply","dist_demand"],
    tp_sl_cols=["tp","sl"],
    epochs=20,
    batch_size=16,
    device="cuda",
    save_best_path="best_model.pt",
    val_split=0.2,
    early_stopping_patience=5,
    log_dir="./logs"
)
```

Documentation Index
Architecture – high‑level design and data flow.

Model – EntryExitTransformer internals.

Data Preparation – feature/label formats and feature engineering.

Training & Self‑training – training loop, metrics, pseudo‑labeling.

DSL – language reference and integration with the model.

Contracts – batch validation rules.

Quick Start – detailed walkthrough of quick_train.

License
[Specify your license]

## Next Steps

Read the detailed documentation for each component:

- [Overview](README.md)
- [Architecture](architecture.md)
- [Model](model.md)
- [Contract](contract.md)
- [Data Preparation](data.md)
- [Training & Self‑training](training.md)
