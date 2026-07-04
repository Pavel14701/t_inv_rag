import itertools

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import TradingDataset, collate_ob
from .datatypes import OrderBlock
from .io import (
    load_features_parquet,
    load_labels_parquet,
    merge_features_labels,
    save_labels_parquet,
)
from .losses import dual_loss


def build_loader_from_parquet(
    features_path: str,
    labels_path: str,
    order_blocks: list[OrderBlock],
    seq_len: int,
    price_cols: list[str],
    ind_cols: list[str],
    sig_cols: list[str],
    tp_sl_cols: list[str],
    batch_size: int,
    shuffle: bool,
) -> tuple[DataLoader, pl.DataFrame]:
    df_feat = load_features_parquet(features_path)
    df_lbl = load_labels_parquet(labels_path)
    df = merge_features_labels(df_feat, df_lbl)

    data = np.column_stack([
        df[price_cols].to_numpy(),
        df[ind_cols].to_numpy() if ind_cols else np.zeros((df.height, 0)),
        df[sig_cols].to_numpy() if sig_cols else np.zeros((df.height, 0)),
        df[tp_sl_cols].to_numpy(),
    ])

    action = df['action'].to_numpy()
    outcome = df['outcome'].to_numpy()

    dataset = TradingDataset(
        data=data,
        order_blocks=order_blocks,
        action_targets=action,
        outcome_targets=outcome,
        seq_len=seq_len,
        price_feats=len(price_cols),
        ind_feats=len(ind_cols),
        sig_feats=len(sig_cols),
        tp_sl_feats=len(tp_sl_cols),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ob
    )
    return loader, df


def train_one_round(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    outcome_mode: str = 'binary',
    lambda_outcome: float = 0.3,
    lr: float = 1e-4,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            prices, indicators, signals, tp, sl, order_blocks, action_targets, outcome_targets = batch  # noqa: E501
            prices = prices.to(device)
            indicators = indicators.to(device)
            signals = signals.to(device)
            tp = tp.to(device)
            sl = sl.to(device)
            action_targets = action_targets.to(device)
            outcome_targets = outcome_targets.to(device)

            optimizer.zero_grad()
            action_logits, outcome_logits = model(
                prices, indicators, signals, tp, sl, order_blocks
            )
            loss, _, _ = dual_loss(
                action_logits,
                outcome_logits,
                action_targets,
                outcome_targets,
                outcome_mode,
                lambda_outcome,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                prices, indicators, signals, tp, sl, order_blocks, action_targets, outcome_targets = batch  # noqa: E501
                prices = prices.to(device)
                indicators = indicators.to(device)
                signals = signals.to(device)
                tp = tp.to(device)
                sl = sl.to(device)
                action_targets = action_targets.to(device)
                outcome_targets = outcome_targets.to(device)

                action_logits, outcome_logits = model(
                    prices, indicators, signals, tp, sl, order_blocks
                )
                loss, _, _ = dual_loss(
                    action_logits,
                    outcome_logits,
                    action_targets,
                    outcome_targets,
                    outcome_mode,
                    lambda_outcome,
                )
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}')  # noqa: E501
        scheduler.step(avg_val_loss)

    return model


# ----------------------------------------------------------------------
# Генерация псевдо-меток для одного батча
# ----------------------------------------------------------------------
def _generate_pseudo_labels_batch(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    outcome_mode: str,
    action_threshold: float,
    outcome_threshold: float,
    min_rr: float,
    close_idx: int,
    batch_idx: int,
    batch_size: int,
    seq_len: int,
) -> list[tuple[int, int, float]]:
    """Обрабатывает один батч и возвращает список псевдо-меток:
    (global_index, pseudo_action, pseudo_outcome).
    """
    prices, indicators, signals, tp, sl, order_blocks_batch, action_targets, outcome_targets = batch
    prices = prices.to(device)
    indicators = indicators.to(device)
    signals = signals.to(device)
    tp = tp.to(device)
    sl = sl.to(device)

    action_logits, outcome_logits = model(prices, indicators, signals, tp, sl, order_blocks_batch)
    action_probs = F.softmax(action_logits, dim=-1)
    max_action_probs, pred_action = action_probs.max(dim=-1)

    B, T = prices.shape[0], prices.shape[1]
    new_pseudo = []

    for b, t in itertools.product(range(B), range(T)):
        if action_targets[b, t] != -100:
            continue

        if max_action_probs[b, t] <= action_threshold or pred_action[b, t] not in (1, 2):
            continue

        # Проверка RR для entry
        if pred_action[b, t] == 1:
            entry_price = prices[b, t, close_idx].item()
            tp_price = tp[b, t, 0].item()
            sl_price = sl[b, t, 0].item()

            if entry_price > sl_price:  # long
                rr = (tp_price - entry_price) / (entry_price - sl_price)
                valid_rr = (sl_price < entry_price < tp_price) and (rr >= min_rr)
            else:  # short
                rr = (entry_price - tp_price) / (sl_price - entry_price)
                valid_rr = (tp_price < entry_price < sl_price) and (rr >= min_rr)

            if not valid_rr:
                continue

        # Определение псевдо-исхода
        if outcome_mode == 'binary':
            outcome_prob = torch.sigmoid(outcome_logits[b, t])
            if outcome_prob > outcome_threshold:
                pseudo_outcome = 1.0
            elif outcome_prob < (1 - outcome_threshold):
                pseudo_outcome = 0.0
            else:
                continue
        elif outcome_mode == 'multiclass':
            probs = F.softmax(outcome_logits[b, t], dim=-1)
            max_prob, cls = probs.max(dim=-1)
            if max_prob < outcome_threshold:
                continue
            pseudo_outcome = float(cls.item())
        elif outcome_mode == 'regression':
            pred_value = outcome_logits[b, t].item()
            if abs(pred_value) >= outcome_threshold:
                pseudo_outcome = float(pred_value)
            else:
                continue
        else:
            continue

        pseudo_action = int(pred_action[b, t].item())
        global_idx = batch_idx * batch_size * T + b * T + t
        new_pseudo.append((global_idx, pseudo_action, pseudo_outcome))

    return new_pseudo


# ----------------------------------------------------------------------
# Обновление parquet-файла с метками
# ----------------------------------------------------------------------
def _update_labels_parquet(
    labels_path: str,
    new_pseudo: list[tuple[int, int, float]],
) -> None:
    """Загружает labels.parquet, обновляет action/outcome по глобальным индексам и сохраняет."""
    df_lbl = load_labels_parquet(labels_path)

    # Если колонок нет, создаём с игнор-значениями
    if 'action' not in df_lbl.columns:
        df_lbl = df_lbl.with_columns([
            pl.lit(-100).alias('action'),
            pl.lit(float('nan')).alias('outcome'),
        ])

    action_arr = df_lbl['action'].to_numpy()
    outcome_arr = df_lbl['outcome'].to_numpy()

    for global_idx, pseudo_action, pseudo_outcome in new_pseudo:
        if 0 <= global_idx < len(action_arr):
            action_arr[global_idx] = pseudo_action
            outcome_arr[global_idx] = pseudo_outcome

    df_lbl = df_lbl.with_columns([
        pl.Series('action', action_arr),
        pl.Series('outcome', outcome_arr),
    ])
    save_labels_parquet(df_lbl, labels_path)


# ----------------------------------------------------------------------
# Основная функция self_training_loop (теперь короткая и читаемая)
# ----------------------------------------------------------------------
def self_training_loop(
    model: torch.nn.Module,
    features_path: str,
    labels_path: str,
    order_blocks: list[OrderBlock],
    price_cols: list[str],
    ind_cols: list[str],
    sig_cols: list[str],
    tp_sl_cols: list[str],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    outcome_mode: str = 'binary',
    lambda_outcome: float = 0.3,
    lr: float = 1e-4,
    epochs_per_round: int = 3,
    num_rounds: int = 3,
    action_threshold: float = 0.9,
    outcome_threshold: float = 0.8,
    min_rr: float = 1 / 3,
    close_idx: int = 3,
):
    labeled_loader, _ = build_loader_from_parquet(
        features_path, labels_path, order_blocks,
        seq_len, price_cols, ind_cols, sig_cols, tp_sl_cols,
        batch_size=batch_size, shuffle=True,
    )
    val_loader = labeled_loader
    unlabeled_loader = labeled_loader

    for round_idx in range(num_rounds):
        print(f'Self-training round {round_idx + 1}/{num_rounds}')

        # Шаг 1: обучаем модель на текущем labelled-наборе
        model = train_one_round(
            model,
            labeled_loader,
            val_loader,
            epochs_per_round,
            device,
            outcome_mode,
            lambda_outcome,
            lr,
        )

        # Шаг 2: генерируем псевдо-метки на unlabelled-данных
        model.eval()
        all_new_pseudo = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(unlabeled_loader):
                batch_pseudo = _generate_pseudo_labels_batch(
                    model, batch, device, outcome_mode,
                    action_threshold, outcome_threshold, min_rr,
                    close_idx, batch_idx, batch_size, seq_len,
                )
                all_new_pseudo.extend(batch_pseudo)
        if not all_new_pseudo:
            print('No new pseudo-labels, stopping self-training.')
            break
        print(
            f'Generated {len(all_new_pseudo)} pseudo-labels. Updating labels.parquet...'
        )
        # Шаг 3: обновляем файл с метками
        _update_labels_parquet(labels_path, all_new_pseudo)
        # Шаг 4: пересоздаём загрузчики с обновлёнными метками
        labeled_loader, _ = build_loader_from_parquet(
            features_path, labels_path, order_blocks,
            seq_len, price_cols, ind_cols, sig_cols, tp_sl_cols,
            batch_size=batch_size, shuffle=True,
        )
        unlabeled_loader = labeled_loader
    return model
