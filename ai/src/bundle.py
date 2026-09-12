"""Model bundle and inference contract for the EntryExitTransformer.

TZ-06 п.2.4 / п.2.5: a ``torch.save(state_dict)`` without the model
config, feature columns, ``atr_global`` and normalisation statistics is
not loadable as a usable predictor — the architecture and data contract
would have to be guessed. This module bundles everything the inference
path (Risk Engine, TZ-05 script) needs into a single artifact and
provides the single inference entry point ``predict_p_win``.
"""

from __future__ import annotations

import dataclasses

from pathlib import Path

import torch

from .transformer import EntryExitTransformer


@dataclasses.dataclass
class ModelBundle:
    """Self-contained saved model artifact.

    Attributes:
        state_dict (dict): Model weights.
        model_config (dict): Constructor kwargs for
            :class:`EntryExitTransformer`.
        feature_columns (list[str] | None): Ordering of the data columns.
        seq_len (int): Temporal window length.
        atr_global (float): Global ATR used to normalise OB zones.
        norm_stats (dict | None): Normalisation statistics.
        version (int): Bundle schema version.

    """

    state_dict: dict
    model_config: dict
    feature_columns: list[str] | None = None
    seq_len: int = 128
    atr_global: float = 1.0
    norm_stats: dict | None = None
    version: int = 1

    @property
    def outcome_mode(self) -> str:
        """The outcome mode from the model config."""
        return self.model_config.get('outcome_mode', 'binary')


def build_bundle(model, *, model_config: dict, **extra) -> ModelBundle:
    """Collect a model plus its metadata into a :class:`ModelBundle`.

    Args:
        model: Trained model whose ``state_dict`` is captured.
        model_config: All constructor kwargs of the model.
        **extra: Additional bundle fields.

    Returns:
        A populated :class:`ModelBundle`.

    """
    config = dict(model_config)
    if 'outcome_mode' not in config:
        config['outcome_mode'] = getattr(model, 'outcome_mode', 'binary')
    if 'atr_global' not in config:
        config['atr_global'] = getattr(model, 'atr_global', 1.0)
    return ModelBundle(
        state_dict=model.state_dict(),
        model_config=config,
        seq_len=extra.get('seq_len', 128),
        atr_global=extra.get('atr_global', config.get('atr_global', 1.0)),
        feature_columns=extra.get('feature_columns'),
        norm_stats=extra.get('norm_stats'),
    )


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    """Persist a bundle to disk as ``.pt``.

    Args:
        bundle: Bundle to save.
        path: Destination path.

    """
    torch.save(dataclasses.asdict(bundle), Path(path))


def load_bundle(path: str | Path) -> ModelBundle:
    """Load a bundle from disk.

    Args:
        path: Path produced by :func:`save_bundle`.

    Returns:
        The deserialized :class:`ModelBundle`.

    """
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        msg = (
            f'{path}: not a ModelBundle payload '
            f'(expected mapping, got {type(payload).__name__})'
        )
        raise TypeError(msg)
    payload.setdefault('version', 1)
    return ModelBundle(**payload)


def rebuild_model(bundle: ModelBundle) -> EntryExitTransformer:
    """Instantiate the model architecture from bundle config.

    Args:
        bundle: Bundle producing a fresh (untrained-weights) model.

    Returns:
        An :class:`EntryExitTransformer` with the bundled architecture.

    """
    cfg = {
        k: v for k, v in bundle.model_config.items() if k != 'outcome_mode'
    }
    return EntryExitTransformer(**cfg)


class EntryExitPredictor:
    """Single-entry-point inference wrapper (TZ-06 п.2.5).

    Encapsulates the model and its bundle so that consumers (Risk Engine,
    TZ-05 script) never touch raw tensors or transformer internals.
    """

    def __init__(self, bundle: ModelBundle):
        """Build the predictor from a bundle.

        Args:
            bundle: Loaded :class:`ModelBundle`.

        """
        self.bundle = bundle
        self.model = rebuild_model(bundle)
        self.model.load_state_dict(bundle.state_dict)
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(
        self,
        prices,
        indicators,
        signals,
        tp_levels,
        sl_levels,
        order_blocks,
    ) -> dict[str, float]:
        """Compute per-decision probabilities on the last bar of a window.

        Args:
            prices: (T, n_price_feats) tensor.
            indicators: (T, n_ind_feats) tensor.
            signals: (T, n_sig_feats) tensor.
            tp_levels: (T, n_tp_sl_feats) tensor.
            sl_levels: (T, n_tp_sl_feats) tensor.
            order_blocks: list of :class:`OrderBlock` in the window.

        Returns:
            Dict with ``p_entry``, ``p_exit`` and ``p_win``.

        """
        device = next(self.model.parameters()).device
        b = lambda x: x.unsqueeze(0).to(device)  # noqa: E731

        action_logits, outcome_logits, _pattern = self.model(
            b(prices), b(indicators), b(signals),
            b(tp_levels), b(sl_levels), [order_blocks],
        )
        action = torch.softmax(action_logits, dim=-1)[0, -1]
        outcome = outcome_logits[0, -1]

        p_entry = float(action[1].item())
        p_exit = float(action[2].item())

        if outcome.shape[-1] == 1:
            p_win = float(torch.sigmoid(outcome[0]).item())
        else:
            p_win = float(
                torch.softmax(outcome, dim=-1)[1].item()
                if outcome.shape[-1] >= 2
                else torch.sigmoid(outcome[0]).item()
            )
        return {'p_entry': p_entry, 'p_exit': p_exit, 'p_win': p_win}

    def predict_p_win(
        self,
        prices,
        indicators,
        signals,
        tp_levels,
        sl_levels,
        order_blocks,
    ) -> float:
        """Compute the probability of a winning outcome on the last bar.

        Args:
            prices: (T, n_price_feats) tensor.
            indicators: (T, n_ind_feats) tensor.
            signals: (T, n_sig_feats) tensor.
            tp_levels: (T, n_tp_sl_feats) tensor.
            sl_levels: (T, n_tp_sl_feats) tensor.
            order_blocks: list of :class:`OrderBlock` in the window.

        Returns:
            Float in ``[0, 1]``.

        """
        return self.predict_proba(
            prices, indicators, signals, tp_levels, sl_levels, order_blocks
        )['p_win']


__all__ = [
    'EntryExitPredictor',
    'ModelBundle',
    'build_bundle',
    'load_bundle',
    'rebuild_model',
    'save_bundle',
]
