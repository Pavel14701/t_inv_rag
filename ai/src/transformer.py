from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OrderBlock


class EntryExitTransformer(nn.Module):
    def __init__(
        self,
        n_price_feats: int,
        n_ind_feats: int,
        n_sig_feats: int,
        n_tp_sl_feats: int = 2,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        max_ob_seq_len: int = 256,
        n_action_classes: int = 3,
        outcome_mode: str = 'binary',
        n_outcome_classes: int = 2,
        n_patterns: int = 10,
        ob_embedding_dim: int = 32,
        atr_global: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.max_ob_seq_len = max_ob_seq_len
        self.outcome_mode = outcome_mode
        self.atr_global = atr_global

        # Временные ряды
        total_time_feats = (
            n_price_feats + n_ind_feats + n_sig_feats + n_tp_sl_feats
        )
        self.time_input_proj = nn.Linear(total_time_feats, hidden_size)
        self.time_pos_encoding = self._positional_encoding(
            max_seq_len,
            hidden_size
        )

        time_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers)

        # Ордерблоки
        self.ob_numeric_proj = nn.Linear(5, hidden_size // 2)  # 5 числовых признаков
        self.ob_type_embedding = nn.Embedding(2, ob_embedding_dim)
        self.ob_structure_embedding = nn.Embedding(4, ob_embedding_dim)
        self.ob_trend_embedding = nn.Embedding(3, ob_embedding_dim)
        self.ob_merge = nn.Linear(hidden_size // 2 + ob_embedding_dim * 3, hidden_size)

        self.ob_pos_encoding = self._positional_encoding(max_ob_seq_len, hidden_size)
        ob_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.ob_encoder = nn.TransformerEncoder(ob_layer, num_layers)
        self.ob_cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_action_classes)
        )
        # Outcome head
        if outcome_mode in {'binary', 'regression'}:
            self.outcome_head = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )
        elif outcome_mode == 'multiclass':
            self.outcome_head = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, n_outcome_classes)
            )
        else:
            raise ValueError('Unknown outcome_mode')

        # Pattern head (multi‑label)
        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_patterns)
        )

    def _positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(
                0, d_model, 2
            ).float() * (
                -torch.log(
                    torch.tensor(10000.0)
                ) / d_model
            )
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def _encode_ob(
        self, 
        ob: OrderBlock, 
        seq_len: int
    ) -> tuple[torch.Tensor, int, int, int]:
        start_norm = ob.start_idx / seq_len
        end_norm = ob.end_idx / seq_len
        low_norm = ob.zone_low / self.atr_global
        high_norm = ob.zone_high / self.atr_global
        strength_norm = ob.strength

        numeric = torch.tensor(
            [start_norm, end_norm, low_norm, high_norm, strength_norm],
            dtype=torch.float32
        )

        type_id = 0 if ob.block_type.lower() == 'supply' else 1
        structure_map = {'valid': 0, 'broken': 1, 'weak': 2, None: 3}
        structure_id = structure_map.get(ob.structure_label, 3)
        trend_map = {'up': 0, 'down': 1, None: 2}
        trend_id = trend_map.get(ob.trend_direction, 2)

        return numeric, type_id, structure_id, trend_id

    def forward(
        self,
        prices: torch.Tensor,
        indicators: torch.Tensor,
        signals: torch.Tensor,
        tp_levels: torch.Tensor,
        sl_levels: torch.Tensor,
        order_blocks: List[List[OrderBlock]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = prices.shape
        device = prices.device

        # 1. Time encoder
        time_feats = torch.cat(
            [prices, indicators, signals, tp_levels, sl_levels], 
            dim=-1
        )
        time_emb = self.time_input_proj(time_feats)
        time_emb = time_emb + self.time_pos_encoding[:, :seq_len, :].to(device)
        time_out = self.time_encoder(time_emb)  # (batch, seq_len, hidden)

        # 2. OB encoder
        ob_emb_list = []
        ob_mask_list = []
        max_len = min(
            max((len(x) for x in order_blocks), default=1), self.max_ob_seq_len
        )

        for obs in order_blocks:
            obs = obs[-max_len:]
            vecs = []
            for ob in obs:
                numeric, type_id, structure_id, trend_id = self._encode_ob(ob, seq_len)
                numeric = numeric.to(device)
                num_proj = F.gelu(self.ob_numeric_proj(numeric.unsqueeze(0)))  # (1, hidden//2)  # noqa: E501

                type_emb = self.ob_type_embedding(torch.tensor(
                    [type_id], 
                    device=device
                ))
                struct_emb = self.ob_structure_embedding(torch.tensor(
                    [structure_id], 
                    device=device
                ))
                trend_emb = self.ob_trend_embedding(torch.tensor(
                    [trend_id], 
                    device=device
                ))

                combined = torch.cat(
                    [num_proj, type_emb, struct_emb, trend_emb], 
                    dim=-1
                )
                ob_vec = F.gelu(self.ob_merge(combined))       # (1, hidden)
                vecs.append(ob_vec)

            if not vecs:
                ob_seq = torch.zeros(0, self.hidden_size, device=device)
            else:
                ob_seq = torch.cat(vecs, dim=0)               # (n_ob, hidden)

            pad = max_len - ob_seq.size(0)
            if pad > 0:
                ob_seq = F.pad(ob_seq, (0, 0, 0, pad))
            ob_emb_list.append(ob_seq)

            mask = torch.ones(max_len, dtype=torch.bool, device=device)
            if pad > 0:
                mask[-pad:] = False
            ob_mask_list.append(mask)

        ob_emb = torch.stack(ob_emb_list, dim=0)  # (batch, max_len, hidden)
        ob_mask = torch.stack(ob_mask_list, dim=0)             # (batch, max_len)

        cls_tokens = self.ob_cls_token.expand(batch, -1, -1)
        ob_emb = torch.cat([cls_tokens, ob_emb], dim=1)  # (batch, max_len+1, hidden)
        cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=device)
        ob_mask = torch.cat([cls_mask, ob_mask], dim=1)

        ob_emb = ob_emb + self.ob_pos_encoding[:, :ob_emb.size(1), :].to(device)
        ob_out = self.ob_encoder(ob_emb, src_key_padding_mask=~ob_mask)
        ob_global = ob_out[:, 0, :]                             # (batch, hidden)

        # 3. Heads
        ob_global_exp = ob_global.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([time_out, ob_global_exp], dim=-1)  # (batch, seq_len, hidden*2)  # noqa: E501

        action_logits = self.action_head(combined)
        outcome_logits = self.outcome_head(combined)
        pattern_logits = self.pattern_head(combined)

        return action_logits, outcome_logits, pattern_logits