import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from . import (
    EntryExitTransformer,
    OrderBlock,
    TradingDataset,
    collate_ob,
    compute_atr,
    compute_ob_distances,
    generate_labels_from_strategy,
    train_one_round,
)


def main():
    total_bars = 10000
    df = pl.DataFrame({
        "open": np.random.randn(total_bars) + 100,
        "high": np.random.randn(total_bars) + 101,
        "low": np.random.randn(total_bars) + 99,
        "close": np.random.randn(total_bars) + 100,
        "volume": np.random.randn(total_bars) + 1000,
        "tp": np.random.randn(total_bars) + 105,
        "sl": np.random.randn(total_bars) + 95,
    })

    order_blocks = [
        OrderBlock(
            id=1,
            block_type="demand",
            start=df["open"].to_list()[0],
            break_=df["open"].to_list()[0],
            retest=df["open"].to_list()[0],
            zone_low=100.0,
            zone_high=105.0,
            strength=1.0,
            structure_label="valid",
            trend_direction="up",
            start_idx=100,
            end_idx=105,
        )
    ]

    atr = compute_atr(df)
    dist_supply, dist_demand, dist_strong = compute_ob_distances(
        df, order_blocks, atr
    )

    df = df.with_columns([
        pl.Series("dist_supply", dist_supply),
        pl.Series("dist_demand", dist_demand),
        pl.Series("dist_strong", dist_strong),
    ])

    action, outcome = generate_labels_from_strategy(df, order_blocks)

    price_cols = ["open", "high", "low", "close", "volume"]
    ind_cols = []          # сюда твои индикаторы
    sig_cols = ["dist_supply", "dist_demand", "dist_strong"]  # + паттерны
    tp_sl_cols = ["tp", "sl"]

    data = np.column_stack([
        df[price_cols].to_numpy(),
        df[ind_cols].to_numpy() if ind_cols else np.zeros((total_bars, 0)),
        df[sig_cols].to_numpy(),
        df[tp_sl_cols].to_numpy(),
    ])

    seq_len = 128
    dataset = TradingDataset(
        data,
        order_blocks,
        action,
        outcome,
        seq_len=seq_len,
        price_feats=len(price_cols),
        ind_feats=len(ind_cols),
        sig_feats=len(sig_cols),
        tp_sl_feats=2,
    )

    train_loader = DataLoader(
        dataset, batch_size=16,
        shuffle=True, collate_fn=collate_ob
    )
    val_loader = DataLoader(
        dataset, batch_size=16,
        shuffle=False, collate_fn=collate_ob
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EntryExitTransformer(
        n_price_feats=len(price_cols),
        n_ind_feats=len(ind_cols),
        n_sig_feats=len(sig_cols),
        n_tp_sl_feats=2,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        outcome_mode="binary",
        atr_global=float(atr.mean()),
    ).to(device)

    train_one_round(
        model,
        train_loader,
        val_loader,
        epochs=3,
        device=device,
        outcome_mode="binary",
        lambda_outcome=0.3,
        lr=1e-4,
    )


if __name__ == "__main__":
    main()
