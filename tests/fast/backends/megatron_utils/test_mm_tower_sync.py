"""Frozen multimodal tower re-send is opted into with --mm-tower-sync, for any tower name (radixark/miles#3159)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import json
from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.megatron_utils.update_weight import update_weight_from_tensor as uwt

_INKLING = "miles_plugins.models.inkling.model.inkling_mm_model_provider"


def _args(mm_tower_sync=None, provider=None, hf_checkpoint=None):
    return Namespace(mm_tower_sync=mm_tower_sync, custom_model_provider_path=provider, hf_checkpoint=hf_checkpoint)


_IN_GATHER_GROUP = object()


def _updater(args, gather_group=_IN_GATHER_GROUP):
    """The method under test touches only these three attributes."""
    updater = object.__new__(uwt.UpdateWeightFromTensor)
    updater.args = args
    updater._mm_tower_cache = None
    updater._ipc_gather_group = gather_group
    return updater


@pytest.fixture
def checkpoint(tmp_path):
    tensors = {
        "model.language_model.layers.0.mlp.weight": torch.ones(2, 2),
        "model.visual.blocks.0.attn.proj.weight": torch.full((2,), 3.0),
        "model.visual.merger.ln_q.weight": torch.full((2,), 4.0),
        "audio.encoder.weight": torch.full((2,), 5.0),
    }
    shards = {
        "model-00001.safetensors": [
            "model.language_model.layers.0.mlp.weight",
            "model.visual.blocks.0.attn.proj.weight",
        ],
        "model-00002.safetensors": ["model.visual.merger.ln_q.weight", "audio.encoder.weight"],
    }
    weight_map = {}
    for shard, keys in shards.items():
        save_file({k: tensors[k] for k in keys}, str(tmp_path / shard))
        weight_map.update({k: shard for k in keys})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tmp_path, tensors


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(uwt, "_warned_implicit_mm_tower_sync", False)


def test_off_by_default():
    assert uwt.mm_tower_names(_args()) == ()
    assert uwt.mm_tower_names(_args(provider="fti.trainers.miles.vision.qwen3_5.model_provider")) == ()
    assert _updater(_args(hf_checkpoint="/nonexistent"))._mm_tower_named_tensors() is None


def test_inkling_provider_implies_its_towers_and_the_flag_overrides():
    assert uwt.mm_tower_names(_args(provider=_INKLING)) == ("visual", "audio")
    assert uwt.mm_tower_names(_args(mm_tower_sync=["visual"], provider=_INKLING)) == ("visual",)
    assert uwt.mm_tower_names(_args(mm_tower_sync=[], provider=_INKLING)) == ()


@pytest.mark.parametrize(
    "key,expected",
    [
        ("model.visual.blocks.0.attn.proj.weight", True),
        ("visual.merger.ln_q.weight", True),
        ("model.visualizer.weight", False),
        ("lm_head.weight", False),
    ],
)
def test_tower_key_match(key, expected):
    assert uwt.is_mm_tower_key(key, ("visual",)) is expected


def test_named_towers_come_from_the_checkpoint_and_are_read_once(checkpoint):
    ckpt_dir, tensors = checkpoint
    updater = _updater(_args(mm_tower_sync=["visual"], hf_checkpoint=str(ckpt_dir)))
    got = updater._mm_tower_named_tensors()
    assert sorted(name for name, _ in got) == [
        "model.visual.blocks.0.attn.proj.weight",
        "model.visual.merger.ln_q.weight",
    ]
    for name, tensor in got:
        assert torch.equal(tensor, tensors[name])
    (ckpt_dir / "model.safetensors.index.json").unlink()
    assert updater._mm_tower_named_tensors() is got


def test_two_towers(checkpoint):
    ckpt_dir, _ = checkpoint
    got = _updater(_args(mm_tower_sync=["visual", "audio"], hf_checkpoint=str(ckpt_dir)))._mm_tower_named_tensors()
    assert sorted(name for name, _ in got) == [
        "audio.encoder.weight",
        "model.visual.blocks.0.attn.proj.weight",
        "model.visual.merger.ln_q.weight",
    ]


def test_ranks_outside_the_ipc_gather_group_contribute_nothing(checkpoint):
    ckpt_dir, _ = checkpoint
    assert (
        _updater(
            _args(mm_tower_sync=["visual"], hf_checkpoint=str(ckpt_dir)), gather_group=None
        )._mm_tower_named_tensors()
        == []
    )
