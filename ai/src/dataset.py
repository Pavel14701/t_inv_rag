"""Trading dataset that produces sliding windows of market data.

Each sample includes price, indicator, signal, TP/SL tensors,
a list of relevant order blocks, action/outcome targets,
optionally pattern targets, the global positional start index,
and a stable bar identifier for safe pseudo-label alignment.
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
        bar_index: Optional 1D array of stable bar identifiers (e.g. a
            timestamp or custom index). If None, the positional index is
            used as the identifier.

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
        bar_index: np.ndarray | None = None,
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

        self.pattern_targets: torch.Tensor | None = None
        if pattern_targets is not None:
            self.pattern_targets = torch.tensor(
                pattern_targets, dtype=torch.float32
            )

        self.bar_index: torch.Tensor | None = None
        if bar_index is not None:
            self.bar_index = torch.tensor(bar_index, dtype=torch.long)

        # TZ-06 item 2.6: order blocks sorted by end_idx so that each window
        # only scans the prefix of blocks that could fall inside it
        # (bisect instead of a full scan over all blocks).
        self._ob_sorted = sorted(self.order_blocks, key=lambda ob: ob.end_idx)
        self._ob_end_idx = np.asarray(
            [ob.end_idx for ob in self._ob_sorted], dtype=np.int64
        )

    def __len__(self) -> int:
        """Return the number of possible sliding windows."""
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx: int):
        """Return a single window sample.

        Args:
            idx: Starting bar index of the window.

        Returns:
            tuple: (prices, indicators, signals, tp, sl, ob_window,
                    action_target, outcome_target, pattern_target,
                    start_bar, bar_idx)

        """
        window = self.data[idx : idx + self.seq_len]
        prices = window[:, : self.price_feats]

        ind_start = self.price_feats
        ind_end = ind_start + self.ind_feats
        indicators = window[:, ind_start:ind_end]

        sig_start = ind_end
        sig_end = sig_start + self.sig_feats
        signals = window[:, sig_start:sig_end]

        tp_sl = window[:, -self.tp_sl_feats :]
        tp = tp_sl[:, 0:1]
        sl = tp_sl[:, 1:2]

        start_bar = idx
        end_bar = idx + self.seq_len - 1
        # blocks with end_idx <= end_bar form a prefix (sorted); among
        # them keep those whose end is not before the window start
        prefix = int(np.searchsorted(self._ob_end_idx, end_bar, side="right"))
        ob_window = [
            ob for ob in self._ob_sorted[:prefix] if ob.end_idx >= start_bar
        ]

        action_target = self.action_targets[idx : idx + self.seq_len]
        outcome_target = self.outcome_targets[idx : idx + self.seq_len]

        if self.pattern_targets is not None:
            pattern_target = self.pattern_targets[idx : idx + self.seq_len]
        else:
            pattern_target = torch.zeros(self.seq_len, 0)

        # stable identifier for the first bar of the window
        bar_idx = self.bar_index[idx] if self.bar_index is not None else idx
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
            start_bar,  # positional index (idx)
            bar_idx,  # stable identifier (or idx if not provided)
        )


def collate_ob(batch):
    """Collate function for DataLoader.

    Stacks all tensors and collects order block lists.
    The batch now contains 11 elements: the 11th is a tensor of
    stable bar identifiers for each window.

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
    start_indices = torch.tensor([item[9] for item in batch], dtype=torch.long)
    bar_indices = torch.tensor([item[10] for item in batch], dtype=torch.long)
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
        bar_indices,
    )
