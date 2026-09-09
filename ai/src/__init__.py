"""Entry-Exit Transformer trading system.

This package provides a full machine-learning pipeline for order-block-based
trade decisions.  The core idea is to combine classical technical features
(price, indicators, signals) with structured information from supply/demand
order blocks, then train a transformer model to predict three things per bar:

* **action** - hold (0), entry (1), exit (2), or ignore (-100)
* **outcome** - win/loss (binary), class label (multiclass), or R-multiple
    (regression)
* **pattern** - multi-label pattern classification (optional, trained only
    when ``pattern_cols`` are provided)

The system supports both supervised learning on pre-labelled data and
semi-supervised *self-training*, where the model iteratively labels an
unlabelled dataset and adds confident predictions to the training set.

Module overview
---------------
**datatypes**
    :class:`OrderBlock` - immutable dataclass representing a
    supply/demand zone.

**features**
    :func:`compute_atr`
        Average True Range calculation.
    :func:`compute_ob_distances`
        Efficient, vectorised computation of ATR-normalised distances to the
        nearest supply zone, demand zone, and the strongest order block.
    :func:`generate_labels_from_strategy`
        A state-machine simulator that walks through bars and produces
        per-bar action (-100/1/2) and outcome labels using order blocks,
        take-profit, stop-loss, and risk-reward filters.

**io**
    Functions for reading and writing Parquet files:
    :func:`load_features_parquet`, :func:`load_labels_parquet`,
    :func:`save_labels_parquet`, :func:`load_order_blocks_parquet`.
    :func:`merge_features_labels` joins feature and label DataFrames,
    filling missing values with appropriate ignore tokens.

**dataset**
    :class:`TradingDataset` - a PyTorch Dataset that slices sliding windows
    from a large 2D array and filters order blocks that fall inside each
    window.  Now optionally returns pattern targets.
    :func:`collate_ob` - collate function for DataLoader that handles
    10 elements per sample (including global bar indices).

**losses**
    :func:`dual_loss` - combines a cross-entropy loss on actions, an
    auxiliary outcome loss applied only to entry bars, and an optional
    multi-label pattern loss.  Supports class weighting for imbalanced
    actions.

**transformer**
    :class:`EntryExitTransformer` - the main model.  It contains:

    * A *time encoder* (TransformerEncoder) that processes the concatenated
        sequence of prices, indicators, signals, TP, and SL.
    * An *order-block encoder* (another TransformerEncoder) that embeds
        numeric and categorical properties of order blocks, prepends a CLS
        token, and produces a global OB representation.
    * Three heads (action, outcome, pattern) that operate on the
        concatenation of the time-encoder output and the tiled OB-global
        vector.

**training**
    :func:`build_loader_from_parquet`
        Creates a DataLoader from labelled Parquet files.  ``pattern_cols``
        can be supplied to include pattern targets.
    :func:`build_unlabeled_loader_from_parquet`
        Creates a DataLoader with all targets set to ignore (patterns empty).
    :func:`train_one_round`
        Standard supervised training loop with AdamW, ReduceLROnPlateau,
        validation metrics, checkpointing, early stopping, and optional
        TensorBoard logging.
    :func:`self_training_loop`
        Iterative self-training: train → pseudo-label → update labels →
        repeat.

**metrics**
    :func:`compute_action_accuracy`
        Per-class and overall accuracy for action predictions.
    :func:`compute_trade_metrics`
        Simulates trades on validation data to estimate win rate and
        profit factor (rough monitoring, not a full backtest).

**contracts**
    :func:`validate_batch` - validates the full 10-element batch structure,
    tensor properties, label ranges, and order block integrity.

**quickstart**
    :func:`quick_train` - one-function entry point for training with
    minimal boilerplate.  See quick-start examples below.

Quick start
-----------
If your data is already prepared as Parquet files, the fastest way to
train a model is via :func:`quick_train`:

.. code-block:: python

    from trading.quickstart import quick_train

    model = quick_train(
        features="data/features.parquet",
        labels="data/labels.parquet",
        order_blocks="data/order_blocks.parquet",
        price_cols=["open", "high", "low", "close", "volume"],
        sig_cols=["dist_supply", "dist_demand"],
        tp_sl_cols=["tp", "sl"],
        epochs=10,
        batch_size=16,
    )

This single call loads the data, builds the model, creates loaders,
trains for the specified number of epochs, and returns the trained
:class:`EntryExitTransformer`.

For more control, follow the step-by-step workflow described below.

Typical usage
-------------
1. Prepare three Parquet files:

   * ``features.parquet`` - columns: *open, high, low, close, volume* (or
        OHLCV), plus indicator columns (e.g. RSI, MACD), signal columns (e.g.
     distances to order blocks), and *tp, sl* (absolute price levels).
   * ``labels.parquet`` - columns: *action* (int) and *outcome* (float),
        aligned with the features rows. Optionally pattern label columns.
   * ``order_blocks.parquet`` - serialised :class:`OrderBlock` objects.

2. Load order blocks:

    >>> from trading import load_order_blocks_parquet
    >>> obs = load_order_blocks_parquet('order_blocks.parquet')

3. (Optional) Generate labels from a mechanical strategy:

    >>> from trading import load_features_parquet, save_labels_parquet
    >>> from trading import generate_labels_from_strategy
    >>> df = load_features_parquet('features.parquet')
    >>> action, outcome = generate_labels_from_strategy(df, obs)
    >>> import polars as pl
    >>> lbl_df = pl.DataFrame({'action': action, 'outcome': outcome})
    >>> save_labels_parquet(lbl_df, 'labels.parquet')

4. Build loaders and train:

    >>> from trading import build_loader_from_parquet, train_one_round
    >>> from trading import EntryExitTransformer
    >>> model = EntryExitTransformer(n_price_feats=5, n_ind_feats=0,
    ...                               n_sig_feats=2)
    >>> train_loader, _ = build_loader_from_parquet(
    ...     'features.parquet', 'labels.parquet', obs,
    ...     seq_len=128, price_cols=['open','high','low','close','volume'],
    ...     ind_cols=[], sig_cols=['dist_supply','dist_demand'],
    ...     tp_sl_cols=['tp','sl'], batch_size=16, shuffle=True,
    ...     pattern_cols=['pattern_1','pattern_2'])  # optional
    >>> val_loader = train_loader  # or a separate validation set
    >>> model = train_one_round(model, train_loader, val_loader,
    ...                         epochs=10, device=device)

5. (Optional) Self-training with an unlabelled features file:

    >>> from trading import self_training_loop
    >>> model = self_training_loop(
    ...     model, 'features.parquet', 'labels.parquet',
    ...     'features_unlabeled.parquet', obs, ...)

Data contract
-------------
All tensors entering the model must:

* Be ``float32`` (``float64`` discouraged, ``float16`` only with AMP).
* Have no NaN or Inf values.
* Have consistent batch and sequence lengths.
* TP/SL must be positive absolute prices.
* Order blocks must have valid ``start_idx``, ``end_idx`` (0 ≤ idx <
    seq_len), ``zone_low < zone_high``, and ``strength ≥ 0``.

Use :func:`validate_batch` during development to catch violations early.

Notes
-----
The pattern head is trained only when ``pattern_cols`` is provided.
Otherwise ``pattern_loss`` is zero and the head outputs are unused.

"""

from .config import (
    AIConfig,
    ComputeConfig,
    ModelConfig,
    RiskConfig,
    TrainingConfig,
    load_config,
    risk_kwargs,
    set_seed,
)
from .contracts import validate_batch
from .dataset import TradingDataset, collate_ob
from .datatypes import OrderBlock
from .device import (
    export_onnx,
    resolve_infer_device,
    resolve_train_device,
)
from .features import (
    compute_atr,
    compute_ob_distances,
    compute_tp_sl,
    generate_labels_from_strategy,
)
from .io import (
    load_features_parquet,
    load_labels_parquet,
    load_order_blocks_parquet,
    merge_features_labels,
    save_labels_parquet,
)
from .losses import dual_loss
from .bundle import (
    EntryExitPredictor,
    ModelBundle,
    build_bundle,
    load_bundle,
    rebuild_model,
    save_bundle,
)
from .metrics import (
    compute_action_accuracy,
    compute_trade_metrics,
)
from .quickstart import quick_train
from .training import (
    build_loader_from_parquet,
    self_training_loop,
    train_one_round,
)
from .transformer import EntryExitTransformer

__all__ = [
    'AIConfig',
    'RiskConfig',
    'ModelConfig',
    'TrainingConfig',
    'ComputeConfig',
    'load_config',
    'set_seed',
    'risk_kwargs',
    'resolve_train_device',
    'resolve_infer_device',
    'export_onnx',
    'OrderBlock',
    'TradingDataset',
    'collate_ob',
    'compute_atr',
    'compute_ob_distances',
    'compute_tp_sl',
    'generate_labels_from_strategy',
    'load_features_parquet',
    'load_labels_parquet',
    'save_labels_parquet',
    'load_order_blocks_parquet',
    'merge_features_labels',
    'dual_loss',
    'EntryExitTransformer',
    'ModelBundle',
    'EntryExitPredictor',
    'build_bundle',
    'save_bundle',
    'load_bundle',
    'rebuild_model',
    'build_loader_from_parquet',
    'train_one_round',
    'self_training_loop',
    'validate_batch',
    'compute_action_accuracy',
    'compute_trade_metrics',
    'quick_train',
]
