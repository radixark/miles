"""A checkpoint initializes a trainer only if its MTP layers fit that trainer.

A trainer builds MTP layers only to train them. One that does needs them in the checkpoint; one that does
not still loads a checkpoint with MTP layers, but never that checkpoint's optimizer state for them.
"""

from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
from megatron.training.checkpointing import get_checkpoint_name

from miles.backends.megatron_utils import checkpoint

TRUNK = "decoder.layers.0.mlp.linear_fc1.weight"
MTP = "mtp.layers.0.enorm.weight"
OPTIMIZER = "optimizer.distributed.dp_group_idx_0.param_state"


def _save(load_dir: Path, keys, *, release=False, checkpoint_format="torch_dcp") -> Path:
    path = Path(get_checkpoint_name(str(load_dir), 0 if release else 5, release, return_base_dir=True))
    path.mkdir(parents=True)
    dcp.save({key: torch.zeros(1) for key in keys}, checkpoint_id=path, no_dist=True)
    if checkpoint_format == "torch_dist":
        save_config(CheckpointingConfig(sharded_backend="torch_dist"), str(path))
    (load_dir / "latest_checkpointed_iteration.txt").write_text("release" if release else "5")
    return load_dir


def _args(load: Path, **overrides) -> Namespace:
    return Namespace(
        **{"load": str(load), "pretrained_checkpoint": None, "finetune": False, "no_load_optim": False}
        | {"ckpt_step": None, "model_name": None, "hf_checkpoint": None, "lora_rank": 0, "lora_adapter_path": None}
        | overrides
    )


def _model(mtp_num_layers):
    return [Namespace(config=Namespace(mtp_num_layers=mtp_num_layers))]


@pytest.mark.parametrize("checkpoint_format", ["torch_dist", "torch_dcp"])
@pytest.mark.parametrize(
    ("saved", "built", "flags", "error"),
    [
        # A converted checkpoint carries the MTP layers for conversion's sake and no optimizer state.
        ([TRUNK, MTP], None, {}, None),
        ([TRUNK, OPTIMIZER], None, {}, None),
        ([TRUNK, MTP, OPTIMIZER], 1, {}, None),
        # Optimizer state saved with MTP layers covers parameters the trainer no longer has.
        ([TRUNK, MTP, OPTIMIZER], None, {}, "holds MTP layers with their optimizer state"),
        # ...unless Megatron does not restore it.
        ([TRUNK, MTP, OPTIMIZER], None, {"no_load_optim": True}, None),
        ([TRUNK, MTP, OPTIMIZER], None, {"finetune": True}, None),
        ([TRUNK, MTP, OPTIMIZER], None, {"release": True}, None),
        # Trained MTP layers would start from random weights.
        ([TRUNK, OPTIMIZER], 1, {}, "holds none, so they would start from random weights"),
        ([TRUNK], 1, {"finetune": True}, "holds none"),
    ],
)
def test_a_checkpoint_fits_the_mtp_layers_of_the_trainer_it_initializes(
    tmp_path, saved, built, flags, error, checkpoint_format
):
    flags = flags.copy()
    release = flags.pop("release", False)
    args = _args(_save(tmp_path / "ckpt", saved, release=release, checkpoint_format=checkpoint_format), **flags)
    if error is None:
        checkpoint._check_mtp_checkpoint(args, _model(built))
    else:
        with pytest.raises(AssertionError, match=error):
            checkpoint._check_mtp_checkpoint(args, _model(built))


def test_load_checkpoint_checks_the_loads_that_initialize_the_trained_model(tmp_path, monkeypatch):
    args = _args(_save(tmp_path / "ckpt", [TRUNK, MTP, OPTIMIZER]))
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    loaded = []
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **kwargs: loaded.append(kwargs) or (5, 0))

    with pytest.raises(AssertionError, match="optimizer state"):
        checkpoint.load_checkpoint(_model(None), object(), None, {}, False)
    assert not loaded

    # A reference or teacher load passes no optimizer and only borrows the trunk's weights.
    checkpoint.load_checkpoint(_model(None), None, None, {}, False)
    assert len(loaded) == 1


def _legacy(tmp_path: Path) -> Path:
    """A torch-format checkpoint: one pickled shard per rank and no tensor metadata."""
    load_dir = tmp_path / "ckpt"
    shard = load_dir / "iter_0000005" / "mp_rank_00" / "model_optim_rng.pt"
    shard.parent.mkdir(parents=True)
    torch.save({"model": {TRUNK: torch.zeros(1)}, "iteration": 5}, shard)
    (load_dir / "latest_checkpointed_iteration.txt").write_text("5")
    return load_dir


@pytest.mark.parametrize("no_load_optim", [False, True])
def test_legacy_checkpoint_reaches_megatrons_loader_without_distributed_metadata(tmp_path, monkeypatch, no_load_optim):
    load_dir = _legacy(tmp_path)
    args = _args(load_dir, no_load_optim=no_load_optim)
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    loaded = []
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **kwargs: loaded.append(kwargs) or (5, 0))

    result = checkpoint.load_checkpoint(_model(None), object(), None, {}, False)

    assert result == (5, 0, False)
    assert len(loaded) == 1
    assert not (load_dir / "iter_0000005" / ".metadata").exists()


def test_a_legacy_checkpoint_cannot_start_mtp_training(tmp_path, monkeypatch):
    """Megatron retries a failed strict legacy load without strict, leaving missing MTP layers at random weights."""
    args = _args(_legacy(tmp_path))
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    loaded = []
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **kwargs: loaded.append(kwargs) or (5, 0))

    with pytest.raises(AssertionError, match="legacy \\(torch format\\) checkpoint that cannot be checked"):
        checkpoint.load_checkpoint(_model(1), object(), None, {}, False)
    assert not loaded
