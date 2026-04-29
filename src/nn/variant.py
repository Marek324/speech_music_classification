# nn/variant.py
# Shared factory for TCN-variant modules (small_tcn, tcn_lstm).
# Each variant is a thin config/weights-path wrapper around the canonical TCN
# pipeline in src/nn/tcn/. This factory builds the variant's config loader,
# Click command group, and SpeechMusicDetector subclass from a small set of
# parameters — eliminating the ~400 LOC of duplication that lived in the two
# variants' cli.py/config.py/model.py trios.

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import click
import torch
from safetensors.torch import load_file

from .evaluation import run_nn_inference
from .tcn.config import (
    _MODEL_KEYS,
    _TOP_KEYS,
    get_preprocess_stats_path,
    get_weights_path,
)
from .tcn.evaluation import eval_tcn
from .tcn.model import SpeechMusicDetector
from .tcn.training import train_tcn
from ..evaluator import run_evaluation

log = logging.getLogger(__name__)

_results_dir = Path(__file__).resolve().parent.parent.parent / "results"


@dataclass(frozen=True)
class Variant:
    name: str
    display_name: str
    ui_label: str
    cls: type
    group: click.Group
    get_config: Callable[..., Dict[str, Any]]
    get_weights_path: Callable[[], Path]
    get_stats_path: Callable[[], Path]
    DEFAULT: Dict[str, Any]


def make_variant(
    *,
    name: str,
    display_name: str,
    ui_label: str | None = None,
    cli_group: str,
    group_help: str,
    weights_stem: str,
    subdir: str,
    config: Dict[str, Any],
) -> Variant:
    """Build a Variant from an already-parsed config dict + a handful of naming/path params.

    Parameters
    ----------
    name         : stable module/output identifier ("small_tcn", "tcn_lstm").
    display_name : class name + log-prefix ("SmallTCN", "TCNLSTM").
    ui_label     : user-facing label for the demo picker ("Small TCN"). Falls back to display_name.
    cli_group    : Click subgroup name ("small-tcn", "tcn-lstm").
    group_help   : docstring shown in `nn --help`.
    weights_stem : suffix after `tcn_` in the weights filename; the canonical
                   ``get_weights_path(name=stem)`` emits ``tcn_{stem}.safetensors``.
    subdir       : subdirectory of ``weights/`` where the variant's files live.
    config       : dict with ``tcn`` and ``dataset`` sections (already parsed from TOML).
    """
    tcn_raw = config.get("tcn", {})
    DEFAULT: Dict[str, Any] = {k: v for k, v in tcn_raw.items() if k != "model"}
    DEFAULT["model"] = dict(tcn_raw.get("model", {}))
    DEFAULT["dataset"] = dict(config.get("dataset", {}))

    def get_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        cfg = deepcopy(DEFAULT)
        if overrides:
            for k in _TOP_KEYS:
                if k in overrides:
                    cfg[k] = overrides[k]
            cfg["model"] = {
                **cfg["model"],
                **{k: overrides[k] for k in _MODEL_KEYS if k in overrides},
            }
        return cfg

    def _weights() -> Path:
        return get_weights_path(name=weights_stem, subdir=subdir)

    def _stats() -> Path:
        return get_preprocess_stats_path(name=weights_stem, subdir=subdir)

    class _Variant(SpeechMusicDetector):
        def __init__(self, sample_rate=None, cfg=None, stats_path=None):
            super().__init__(
                sample_rate=sample_rate,
                cfg=cfg if cfg is not None else get_config(),
                stats_path=stats_path if stats_path is not None else _stats(),
            )

    _Variant.__name__ = display_name
    _Variant.__qualname__ = display_name

    @click.group(cli_group, help=group_help)
    def group():
        pass

    @group.command("train", help=f"Train {display_name}.")
    @click.option("--epochs", "-e", default=50, help="Max training epochs")
    @click.option("--patience", "-p", default=5, help="Early-stopping patience (val checks without improvement)")
    @click.option("--no-wandb", is_flag=True, default=False, help="Disable W&B logging")
    def train_cmd(epochs, patience, no_wandb):
        train_tcn(
            epochs=epochs,
            patience=patience,
            cfg=get_config(),
            weights_path=_weights(),
            stats_path=_stats(),
            use_wandb=not no_wandb,
        )

    @group.command("eval", help=f"Evaluate {display_name} on the test split. Writes results/{name}.eval.")
    def eval_cmd():
        eval_tcn(
            cfg=get_config(),
            weights_path=_weights(),
            stats_path=_stats(),
            output_name=name,
            output_dir=_results_dir,
        )

    @group.command("smoke-test")
    def smoke_cmd():
        """Smoke test: forward pass with synthetic audio (no weights or dataset needed)."""
        cfg = get_config()
        model = _Variant(cfg=cfg, stats_path=Path("/nonexistent"))
        nf = model.fe.n_features
        model.fe.norm_mean = torch.zeros(1, nf, 1)
        model.fe.norm_std = torch.ones(1, nf, 1)
        model.fe._stats_loaded = True

        dummy = torch.randn(2, cfg["sample_rate"] * 3)
        probs = model(dummy)

        assert probs.shape[0] == 2 and probs.shape[1] == cfg["model"]["n_classes"], (
            f"Unexpected output shape: {probs.shape}"
        )
        assert 0 <= probs.min().item() <= 1 and 0 <= probs.max().item() <= 1
        n_params = sum(p.numel() for p in model.parameters())
        log.info(
            "%s smoke-test passed. Frontend: %s  Preprocessor: %s  n_filters: %d  n_features: %d  Params: %d  Output shape: %s",
            display_name,
            cfg["frontend"],
            cfg["model"]["preprocessor"],
            cfg["model"]["n_filters"],
            nf,
            n_params,
            tuple(probs.shape),
        )

    @group.command("smoke-test-online")
    def smoke_online_cmd():
        """Train on 10 rows, eval on 3 — verifies full pipeline. Does not overwrite real weights."""
        smoke_weights = _weights().parent / f"{name}.smoke.safetensors"
        log.info("%s smoke-test-online: training on 10 rows (saving to %s)...", display_name, smoke_weights)
        train_tcn(
            epochs=2,
            max_train_rows=10,
            weights_path=smoke_weights,
            use_wandb=False,
            cfg=get_config(),
            stats_path=_stats(),
        )

        log.info("%s smoke-test-online: evaluating on 3 test rows...", display_name)
        cfg = get_config()
        model = _Variant(cfg=cfg, stats_path=_stats())
        model.load_state_dict(load_file(str(smoke_weights), device="cpu"))
        y_true, y_pred, y_sub, tpf_ns, _dev, y_scores, clip_ids = run_nn_inference(
            model, cfg["dataset"], cfg, max_rows=3
        )
        run_evaluation(
            y_true,
            y_pred,
            y_sub,
            tpf_ns,
            output_name=name,
            save_to_file=False,
            y_scores=y_scores,
            clip_ids=clip_ids,
        )
        log.info("%s smoke-test-online passed.", display_name)

    return Variant(
        name=name,
        display_name=display_name,
        ui_label=ui_label or display_name,
        cls=_Variant,
        group=group,
        get_config=get_config,
        get_weights_path=_weights,
        get_stats_path=_stats,
        DEFAULT=DEFAULT,
    )
