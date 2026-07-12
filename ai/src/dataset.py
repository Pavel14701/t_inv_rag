"""Trading dataset that produces sliding windows of market data.

Each sample includes price, indicator, signal, TP/SL tensors,
a list of relevant order blocks, action/outcome targets, and
optionally pattern targets, plus the global start index.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .datatypes import OrderBlock


class TradingDataset(Dataset):
    """Sliding-window dataset for order-block based trading.

    The underlying data array is assumed to have the following layout
    (last dimension):
        [price_feats | ind_feats | sig_feats | tp_sl_feats]

    Args:
        data: 2D float array of shape (total_bars, total_feats).
        order_blocks: All order blocks available for filtering.
        action_targets: 1D array of action labels (hold/entry/exit/ignore).
        outcome_targets: 1D array of outcome labels (win/loss/R-multiple).
        seq_len: Number of bars in each window.
        price_feats: Number of price columns (e.g. 5 for OHLCV).
        ind_feats: Number of indicator columns.
        sig_feats: Number of signal columns.
        tp_sl_feats: Number of TP/SL columns (usually 2: tp, sl).
        pattern_targets: Optional 2D array (total_bars, n_patterns)
            with multi-label pattern indicators. If None, a zero-sized
            tensor is returned for each window.

    """

    def __init__(
        self,
        data: np.ndarray,
        order_blocks: list[OrderBlock],
        action_targets: np.ndarray,
        outcome_targets: np.ndarray,
        seq_len: int = 128,
        price_feats: int = 5,
        ind_feats: int = 3,
        sig_feats: int = 2,
        tp_sl_feats: int = 2,
        pattern_targets: np.ndarray | None = None,
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.order_blocks = order_blocks
        self.action_targets = torch.tensor(action_targets, dtype=torch.long)
        self.outcome_targets = torch.tensor(
            outcome_targets, dtype=torch.float32
        )
        self.seq_len = seq_len
        self.price_feats = price_feats
        self.ind_feats = ind_feats
        self.sig_feats = sig_feats
        self.tp_sl_feats = tp_sl_feats

        # Explicitly declare the type to avoid type-checker confusion
        self.pattern_targets: torch.Tensor | None = None
        if pattern_targets is not None:
            self.pattern_targets = torch.tensor(
                pattern_targets, dtype=torch.float32
            )

    def __len__(self) -> int:
        """Return the number of possible windows."""
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        """Return a single window sample.

        Args:
            idx: Starting bar index of the window.

        Returns:
            tuple: (prices, indicators, signals, tp, sl, ob_window,
                    action_target, outcome_target, pattern_target,
                    start_bar)

        """
        window = self.data[idx: idx + self.seq_len]
        prices = window[:, : self.price_feats]

        ind_start = self.price_feats
        ind_end = ind_start + self.ind_feats
        indicators = window[:, ind_start:ind_end]

        sig_start = ind_end
        sig_end = sig_start + self.sig_feats
        signals = window[:, sig_start:sig_end]

        tp_sl = window[:, -self.tp_sl_feats:]
        tp = tp_sl[:, 0:1]
        sl = tp_sl[:, 1:2]

        start_bar = idx
        end_bar = idx + self.seq_len - 1
        ob_window = [
            ob
            for ob in self.order_blocks
            if start_bar <= ob.end_idx <= end_bar
        ]

        action_target = self.action_targets[idx: idx + self.seq_len]
        outcome_target = self.outcome_targets[idx: idx + self.seq_len]

        if self.pattern_targets is not None:
            pattern_target = self.pattern_targets[
                idx: idx + self.seq_len
            ]
        else:
            pattern_target = torch.zeros(self.seq_len, 0)

        return (
            prices,
            indicators,
            signals,
            tp,
            sl,
            ob_window,
            action_target,
            outcome_target,
            pattern_target,
            start_bar,
        )


def collate_ob(batch):
    """Collate function for DataLoader.

    Stacks all tensors and collects order block lists.
    The batch now contains 10 elements: the 10th is a tensor of
    global start indices for each window.

    Args:
        batch: List of samples as returned by TradingDataset.__getitem__.

    Returns:
        tuple: Collated batch ready for model input.

    """
    prices = torch.stack([item[0] for item in batch])
    indicators = torch.stack([item[1] for item in batch])
    signals = torch.stack([item[2] for item in batch])
    tp = torch.stack([item[3] for item in batch])
    sl = torch.stack([item[4] for item in batch])
    order_blocks = [item[5] for item in batch]
    action_targets = torch.stack([item[6] for item in batch])
    outcome_targets = torch.stack([item[7] for item in batch])
    pattern_targets = torch.stack([item[8] for item in batch])
    start_indices = torch.tensor(
        [item[9] for item in batch], dtype=torch.long
    )
    return (
        prices,
        indicators,
        signals,
        tp,
        sl,
        order_blocks,
        action_targets,
        outcome_targets,
        pattern_targets,
        start_indices,
    )
