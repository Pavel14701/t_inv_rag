"""Strict validation contract for the EntryExitTransformer model.

Performs checks on tensor shapes, dtypes, finite values,
label ranges, and order block integrity before training.
"""

import logging

import torch

from .datatypes import OrderBlock

logger = logging.getLogger(__name__)


def _check_tensor(
    t: torch.Tensor,
    name: str,
    expected_dims: int,
    allowed_dtypes: tuple[torch.dtype, ...] = (
        torch.float32,
        torch.float64,
        torch.int64,
    ),
):
    """Verify a tensor's basic properties.

    Args:
        t: Tensor to check.
        name: Human-readable name used in error messages.
        expected_dims: Expected number of dimensions.
        allowed_dtypes: Tuple of acceptable dtypes.
            Defaults to float32 and float64.

    Raises:
        TypeError: If ``t`` is not a ``torch.Tensor`` or its dtype is
            not allowed.
        ValueError: If ``t`` has incorrect ndim or contains NaN/Inf values.

    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f'{name}: expected torch.Tensor, got {type(t)}')
    if t.ndim != expected_dims:
        raise ValueError(
            f'{name}: expected {expected_dims}D tensor, '
            f'got {t.ndim}D'
        )
    if t.dtype not in allowed_dtypes:
        raise TypeError(
            f'{name}: dtype must be one of {allowed_dtypes}, '
            f'got {t.dtype}. float32 is recommended.'
        )
    if not torch.isfinite(t).all():
        raise ValueError(f'{name}: contains NaN or Inf')


def validate_prices(prices: torch.Tensor, n_price_feats: int):
    """Validate the price tensor (B, T, n_price_feats).

    Args:
        prices: Price tensor of shape (batch, seq_len, n_price_feats).
        n_price_feats: Expected number of price features (e.g., 5 for OHLCV).

    Raises:
        ValueError: If the last dimension does not match ``n_price_feats``.
        TypeError: From ``_check_tensor`` if dtype/dim is wrong.
        ValueError: From ``_check_tensor`` if NaN/Inf is found.

    Warns:
        If any price value <= 0, indicating possible missing normalization.

    """
    _check_tensor(prices, 'prices', 3)
    if prices.shape[-1] != n_price_feats:
        raise ValueError(
            f'prices: last dim must be {n_price_feats}, '
            f'got {prices.shape[-1]}'
        )
    if prices.min() <= 0:
        logger.warning(
            'prices contain non-positive values. Ensure proper '
            'normalization (e.g., z-score or ATR).'
        )


def validate_indicators(indicators: torch.Tensor, n_ind_feats: int):
    """Validate the indicators tensor (B, T, n_ind_feats).

    An empty feature dimension (n_ind_feats=0) is allowed for models
    that do not use indicators.

    Args:
        indicators: Indicator tensor of shape (batch, seq_len, n_ind_feats).
        n_ind_feats: Expected number of indicator features.

    Raises:
        ValueError: If ``n_ind_feats > 0`` and the last
            dimension does not match.
        TypeError/ValueError: From ``_check_tensor`` for dtype/dim/NaN issues.

    Warns:
        If ``n_ind_feats=0`` but the tensor has non-zero feature size.

    """
    _check_tensor(indicators, 'indicators', 3)
    if n_ind_feats > 0:
        if indicators.shape[-1] != n_ind_feats:
            raise ValueError(
                f'indicators: last dim must be {n_ind_feats}, '
                f'got {indicators.shape[-1]}'
            )
    elif indicators.shape[-1] != 0:
        logger.warning(
            'n_ind_feats=0 but indicators has %d features; '
            'they will be ignored by the model.',
            indicators.shape[-1],
        )


def validate_signals(signals: torch.Tensor, n_sig_feats: int):
    """Validate the signals tensor (B, T, n_sig_feats).

    Similar to indicators, zero features are acceptable.

    Args:
        signals: Signal tensor of shape (batch, seq_len, n_sig_feats).
        n_sig_feats: Expected number of signal features.

    Raises:
        ValueError: If ``n_sig_feats > 0`` and dimensions mismatch.
        TypeError/ValueError: From ``_check_tensor``.

    Warns:
        If ``n_sig_feats=0`` but the tensor has non-zero feature size.

    """
    _check_tensor(signals, 'signals', 3)
    if n_sig_feats > 0:
        if signals.shape[-1] != n_sig_feats:
            raise ValueError(
                f'signals: last dim must be {n_sig_feats}, '
                f'got {signals.shape[-1]}'
            )
    elif signals.shape[-1] != 0:
        logger.warning(
            'n_sig_feats=0 but signals has %d features.',
            signals.shape[-1],
        )


def validate_tp_sl(tp: torch.Tensor, sl: torch.Tensor):
    """Validate take-profit and stop-loss tensors.

    Both must be of shape (B, T, 1) and contain positive absolute prices.

    Args:
        tp: Take-profit tensor.
        sl: Stop-loss tensor.

    Raises:
        ValueError: If shapes do not match, last dim != 1, or values <= 0.
        TypeError/ValueError: From ``_check_tensor``.

    """
    for name, t in [('tp', tp), ('sl', sl)]:
        _check_tensor(t, name, 3)
        if t.shape[-1] != 1:
            raise ValueError(
                f'{name}: last dim must be 1, got {t.shape[-1]}'
            )
    if tp.shape != sl.shape:
        raise ValueError(
            f'tp and sl shapes mismatch: {tp.shape} vs {sl.shape}'
        )
    if tp.min() <= 0 or sl.min() <= 0:
        raise ValueError(
            'TP and SL must be positive absolute prices.'
        )


def _validate_single_ob(ob: OrderBlock, seq_len: int, locator: str):
    """Validate a single OrderBlock instance.

    Checks indices, zone boundaries, and strength.

    Args:
        ob: The OrderBlock to validate.
        seq_len: Sequence length for index bounds.
        locator: String like 'order_blocks[{b}][{i}]' for error messages.

    Raises:
        ValueError: If any field is invalid.

    """
    if ob.start_idx < 0 or ob.end_idx < 0:
        raise ValueError(
            f'OrderBlock {ob.id} at {locator}: start_idx/end_idx '
            'must be non-negative'
        )
    if ob.end_idx >= seq_len:
        raise ValueError(
            f'OrderBlock {ob.id} at {locator}: end_idx ({ob.end_idx}) '
            f'exceeds seq_len-1 ({seq_len - 1})'
        )
    if ob.zone_low >= ob.zone_high:
        raise ValueError(
            f'OrderBlock {ob.id} at {locator}: zone_low must be < zone_high'
        )
    if ob.strength < 0:
        raise ValueError(
            f'OrderBlock {ob.id} at {locator}: strength must be >= 0'
        )


def validate_order_blocks(
    order_blocks_list: list[list[OrderBlock]],
    expected_batch_size: int,
    seq_len: int,
):
    """Verify the structure and fields of order blocks for a batch.

    Each batch element must be a list of ``OrderBlock`` objects.
    Individual fields (indices, zone boundaries, strength) are checked
    for validity.

    Args:
        order_blocks_list: List of length B; each element is a list of
            ``OrderBlock`` instances present in the corresponding window.
        expected_batch_size: Expected batch size B.
        seq_len: Sequence length (number of time steps) used for
            index range validation.

    Raises:
        TypeError: If the outer structure is not a list of lists of
            ``OrderBlock``.
        ValueError: If batch size mismatches, or any order block has
            invalid indices, zone_low >= zone_high, or negative strength.

    """
    if not isinstance(order_blocks_list, list):
        raise TypeError('order_blocks must be a list of lists')
    if len(order_blocks_list) != expected_batch_size:
        raise ValueError(
            f'order_blocks batch size {len(order_blocks_list)} '
            f'!= {expected_batch_size}'
        )

    for b, obs in enumerate(order_blocks_list):
        if not isinstance(obs, list):
            raise TypeError(
                f'order_blocks[{b}] must be a list of OrderBlock'
            )
        for i, ob in enumerate(obs):
            if not isinstance(ob, OrderBlock):
                raise TypeError(
                    f'order_blocks[{b}][{i}] is not an OrderBlock'
                )
            _validate_single_ob(ob, seq_len, f'order_blocks[{b}][{i}]')


def validate_action_targets(action_targets: torch.Tensor):
    """Validate action labels: must be -100, 0, 1, or 2.

    -100 is the ignore index used in loss computation; 0,1,2 correspond
    to hold, entry, exit.

    Args:
        action_targets: Tensor of shape (B, T) with integer labels.

    Raises:
        ValueError: If any value is outside the allowed set.
        TypeError: If not a tensor or wrong ndim.

    """
    if not isinstance(action_targets, torch.Tensor):
        raise TypeError(
            'action_targets: expected torch.Tensor, ',
            f'got {type(action_targets)}'
        )
    if action_targets.ndim != 2:
        raise ValueError(
            'action_targets: expected 2D tensor, ',
            f'got {action_targets.ndim}D'
        )
    allowed = {-100, 0, 1, 2}
    unique = action_targets.unique().tolist()
    if invalid := [v for v in unique if v not in allowed]:
        raise ValueError(
            f'action_targets contains invalid values: {invalid}. '
            f'Allowed: {allowed}'
        )


def validate_outcome_targets(
    outcome_targets: torch.Tensor, outcome_mode: str
):
    """Validate outcome labels based on the prediction mode.

    For 'binary' and 'multiclass' modes, finite values must be 0, 1, or 2
    (2 signals "ignore" in the loss). NaN is allowed.
    For 'regression' mode, only a warning is issued for large absolute values.

    Args:
        outcome_targets: Tensor of shape (B, T) with float outcomes.
        outcome_mode: One of 'binary', 'multiclass', 'regression'.

    Raises:
        ValueError: If finite values in classification modes are invalid.
        TypeError/ValueError: From ``_check_tensor``.

    Warns:
        If regression targets contain values with absolute magnitude > 1e6.

    """
    _check_tensor(outcome_targets, 'outcome_targets', 2)
    if outcome_mode in {'binary', 'multiclass'}:
        valid_values = {0.0, 1.0, 2.0}
        finite_mask = torch.isfinite(outcome_targets)
        unique_finite = outcome_targets[finite_mask].unique().tolist()
        if invalid := [v for v in unique_finite if v not in valid_values]:
            raise ValueError(
                f'outcome_targets contains invalid finite values: '
                f'{invalid}. Allowed: {valid_values} or NaN.'
            )
    elif outcome_mode == 'regression':
        if outcome_targets.abs().max() > 1e6:
            logger.warning(
                'outcome_targets has very large values; consider '
                'normalization.'
            )


def validate_batch(
    batch,
    expected_batch_size: int,
    seq_len: int,
    n_price_feats: int,
    n_ind_feats: int,
    n_sig_feats: int,
    outcome_mode: str,
):
    """Run all validations on a single batch from the DataLoader.

    The batch is expected to be a tuple of 11 elements:
    (prices, indicators, signals, tp, sl, order_blocks,
        action_targets, outcome_targets, pattern_targets, start_indices,
        bar_indices).

    Args:
        batch: Tuple of tensors and lists as returned by ``collate_ob``.
        expected_batch_size: Expected batch size (B).
        seq_len: Sequence length (T).
        n_price_feats: Number of price features.
        n_ind_feats: Number of indicator features.
        n_sig_feats: Number of signal features.
        outcome_mode: Outcome prediction mode.

    Raises:
        ValueError: If batch length != 11, shapes mismatch, etc.
        TypeError: From component validators.

    """
    if len(batch) != 11:
        raise ValueError(
            f'Expected batch of 11 elements, got {len(batch)}'
        )
    (
        prices,
        indicators,
        signals,
        tp,
        sl,
        order_blocks,
        action_tgt,
        outcome_tgt,
        pattern_tgt,
        start_indices,
        bar_indices,
    ) = batch
    validate_prices(prices, n_price_feats)
    validate_indicators(indicators, n_ind_feats)
    validate_signals(signals, n_sig_feats)
    validate_tp_sl(tp, sl)
    validate_order_blocks(order_blocks, expected_batch_size, seq_len)
    validate_action_targets(action_tgt)
    validate_outcome_targets(outcome_tgt, outcome_mode)

    b = prices.shape[0]
    t = prices.shape[1]
    if t != seq_len:
        raise ValueError(
            f'Sequence length mismatch: expected {seq_len}, got {t}'
        )
    for name, tensor in [
        ('indicators', indicators),
        ('signals', signals),
        ('tp', tp),
        ('sl', sl),
        ('action_targets', action_tgt),
        ('outcome_targets', outcome_tgt),
        ('pattern_targets', pattern_tgt),
    ]:
        if tensor.shape[0] != b or tensor.shape[1] != t:
            raise ValueError(
                f'{name} shape {tensor.shape} inconsistent '
                f'with (B={b}, T={t})'
            )
    if start_indices.shape != (b,):
        raise ValueError(
            f'start_indices shape {start_indices.shape} != ({b},)'
        )
    if bar_indices.shape != (b,):
        raise ValueError(
            f'bar_indices shape {bar_indices.shape} != ({b},)'
        )
    if pattern_tgt.dtype not in (torch.float32, torch.float64):
        logger.warning(
            'pattern_targets should be float32; got %s.',
            pattern_tgt.dtype,
        )
    if prices.dtype == torch.float64:
        logger.warning(
            'prices are float64; consider converting to float32 '
            'for performance.'
        )
    if prices.dtype == torch.float16:
        logger.warning(
            'float16 detected. Use mixed precision (torch.amp), '
            'otherwise instability may occur.'
        )
    logger.debug('Batch validation passed.')
