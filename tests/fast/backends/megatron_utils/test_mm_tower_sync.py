"""Frozen multimodal tower re-send is opted into with --mm-tower-sync, for any tower name."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import json
from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.megatron_utils.update_weight import hf_weight_iterator as hwi

_INKLING = "miles_plugins.models.inkling.model.inkling_mm_model_provider"


def _args(mm_tower_sync=None, provider=None, hf_checkpoint=None):
    return Namespace(mm_tower_sync=mm_tower_sync, custom_model_provider_path=provider, hf_checkpoint=hf_checkpoint)


@pytest.fixture
def checkpoint(tmp_path):
    """A two-shard HF checkpoint: language weights plus a `visual` tower and an `audio` tower."""
    tensors = {
        "model.language_model.layers.0.mlp.weight": torch.ones(2, 2),
        "lm_head.weight": torch.ones(2),
        "model.visual.blocks.0.attn.proj.weight": torch.full((2,), 3.0),
        "model.visual.merger.ln_q.weight": torch.full((2,), 4.0),
        "audio.encoder.weight": torch.full((2,), 5.0),
    }
    shards = {
        "model-00001.safetensors": [
            "model.language_model.layers.0.mlp.weight",
            "model.visual.blocks.0.attn.proj.weight",
        ],
        "model-00002.safetensors": ["lm_head.weight", "model.visual.merger.ln_q.weight", "audio.encoder.weight"],
    }
    weight_map = {}
    for shard, keys in shards.items():
        save_file({k: tensors[k] for k in keys}, str(tmp_path / shard))
        weight_map.update({k: shard for k in keys})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tmp_path, tensors


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(hwi, "_MM_TOWER_CACHE", None)
    monkeypatch.setattr(hwi, "_warned_implicit_mm_tower_sync", False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


class TestTowerNames:
    def test_off_by_default(self):
        assert hwi.mm_tower_names(_args()) == ()
        assert hwi.mm_tower_names(_args(provider="my_pkg.vlm_provider")) == ()

    def test_flag_names_the_towers(self):
        assert hwi.mm_tower_names(_args(mm_tower_sync=["visual"])) == ("visual",)
        assert hwi.mm_tower_names(_args(mm_tower_sync=["vision_tower", "audio"])) == ("vision_tower", "audio")

    def test_inkling_provider_implies_its_towers(self, caplog):
        with caplog.at_level("WARNING"):
            assert hwi.mm_tower_names(_args(provider=_INKLING)) == ("visual", "audio")
            hwi.mm_tower_names(_args(provider=_INKLING))
        assert sum("--mm-tower-sync visual audio" in r.message for r in caplog.records) == 1

    def test_explicit_flag_overrides_the_inkling_default(self):
        assert hwi.mm_tower_names(_args(mm_tower_sync=["visual"], provider=_INKLING)) == ("visual",)
        # An empty flag value switches the implicit sync off.
        assert hwi.mm_tower_names(_args(mm_tower_sync=[], provider=_INKLING)) == ()


class TestTowerKeys:
    @pytest.mark.parametrize(
        "key",
        ["model.visual.blocks.0.attn.proj.weight", "visual.merger.ln_q.weight", "language_model.visual.x"],
    )
    def test_matches_the_tower_as_a_dotted_module(self, key):
        assert hwi.is_mm_tower_key(key, ("visual",))

    @pytest.mark.parametrize("key", ["model.visualizer.weight", "model.layers.0.visual_gate", "lm_head.weight"])
    def test_does_not_match_a_prefix_of_another_name(self, key):
        assert not hwi.is_mm_tower_key(key, ("visual",))


class TestIterUnits:
    def test_off_yields_nothing_and_reads_no_checkpoint(self, checkpoint):
        ckpt_dir, _ = checkpoint
        args = _args(hf_checkpoint=str(ckpt_dir / "does-not-exist"))
        assert list(hwi._iter_mm_tower_units(args, materialize=True)) == []

    def test_non_materializing_ranks_yield_nothing(self, checkpoint):
        ckpt_dir, _ = checkpoint
        args = _args(mm_tower_sync=["visual"], hf_checkpoint=str(ckpt_dir))
        assert list(hwi._iter_mm_tower_units(args, materialize=False)) == []
        assert hwi._MM_TOWER_CACHE is None

    def test_yields_exactly_the_named_towers_from_the_checkpoint(self, checkpoint):
        ckpt_dir, tensors = checkpoint
        args = _args(mm_tower_sync=["visual"], hf_checkpoint=str(ckpt_dir))
        units = list(hwi._iter_mm_tower_units(args, materialize=True))
        names = [name for unit in units for name, _ in unit]
        assert sorted(names) == ["model.visual.blocks.0.attn.proj.weight", "model.visual.merger.ln_q.weight"]
        for unit in units:
            ((name, tensor),) = unit
            assert torch.equal(tensor, tensors[name])

    def test_two_towers(self, checkpoint):
        ckpt_dir, _ = checkpoint
        args = _args(mm_tower_sync=["visual", "audio"], hf_checkpoint=str(ckpt_dir))
        names = [name for unit in hwi._iter_mm_tower_units(args, materialize=True) for name, _ in unit]
        # Grouped by shard so each shard is opened once; order within the run is not a contract.
        assert sorted(names) == [
            "audio.encoder.weight",
            "model.visual.blocks.0.attn.proj.weight",
            "model.visual.merger.ln_q.weight",
        ]

    def test_checkpoint_is_read_once(self, checkpoint, monkeypatch):
        ckpt_dir, _ = checkpoint
        args = _args(mm_tower_sync=["visual"], hf_checkpoint=str(ckpt_dir))
        first = [name for unit in hwi._iter_mm_tower_units(args, materialize=True) for name, _ in unit]
        (ckpt_dir / "model.safetensors.index.json").unlink()
        second = [name for unit in hwi._iter_mm_tower_units(args, materialize=True) for name, _ in unit]
        assert first == second
