import numpy as np
import torch
from torch.utils.data import Dataset

from .datatypes import OrderBlock


class TradingDataset(Dataset):
    """
    data: (total_bars, total_feats) в порядке:
        [price_feats][ind_feats][sig_feats][tp_sl_feats]
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
    ):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.order_blocks = order_blocks
        self.action_targets = torch.tensor(action_targets, dtype=torch.long)
        self.outcome_targets = torch.tensor(outcome_targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.price_feats = price_feats
        self.ind_feats = ind_feats
        self.sig_feats = sig_feats
        self.tp_sl_feats = tp_sl_feats

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int):
        window = self.data[idx : idx + self.seq_len]
        prices = window[:, : self.price_feats]
        indicators = window[:, self.price_feats : self.price_feats + self.ind_feats]
        signals = window[
            :,
            self.price_feats + self.ind_feats : self.price_feats + self.ind_feats + self.sig_feats,  # noqa: E501
        ]
        tp_sl = window[:, -self.tp_sl_feats :]
        tp = tp_sl[:, 0:1]
        sl = tp_sl[:, 1:2]

        start_bar = idx
        end_bar = idx + self.seq_len - 1
        ob_window = [
            ob for ob in self.order_blocks if start_bar <= ob.end_idx <= end_bar
        ]

        action_target = self.action_targets[idx : idx + self.seq_len]
        outcome_target = self.outcome_targets[idx : idx + self.seq_len]

        return prices, indicators, signals, tp, sl, ob_window, action_target, outcome_target  # noqa: E501


def collate_ob(batch):
    prices = torch.stack([item[0] for item in batch])
    indicators = torch.stack([item[1] for item in batch])
    signals = torch.stack([item[2] for item in batch])
    tp = torch.stack([item[3] for item in batch])
    sl = torch.stack([item[4] for item in batch])
    order_blocks = [item[5] for item in batch]
    action_targets = torch.stack([item[6] for item in batch])
    outcome_targets = torch.stack([item[7] for item in batch])
    return prices, indicators, signals, tp, sl, order_blocks, action_targets, outcome_targets  # noqa: E501
