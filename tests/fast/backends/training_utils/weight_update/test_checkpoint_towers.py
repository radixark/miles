"""Tower tensors a text-only trainer never holds are streamed from the checkpoint."""

import json
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from miles.backends.training_utils.weight_update.hf_weight_iterator import checkpoint_towers

_MODULE = "miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers"


def _checkpoint(tmp_path, *, indexed: bool):
    tensors = {
        "model.layers.0.weight": torch.ones(2),
        "visual.blocks.0.norm1.weight": torch.full((3,), 2.0),
        "model.visual.merger.bias": torch.full((1,), 3.0),
        "audio.encoder.weight": torch.full((2,), 4.0),
    }
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}})
        )
    return str(tmp_path)


def _names(units):
    return sorted(name for unit in units for name, _ in unit)


def test_only_tower_keys_stream_with_or_without_an_index(tmp_path):
    for indexed in (True, False):
        root = tmp_path / ("indexed" if indexed else "flat")
        root.mkdir()
        ckpt = _checkpoint(root, indexed=indexed)
        with patch(f"{_MODULE}.torch.cuda.current_device", return_value="cpu"):
            units = list(checkpoint_towers.iter_checkpoint_tower_units(ckpt, materialize=True))
        assert _names(units) == ["audio.encoder.weight", "model.visual.merger.bias", "visual.blocks.0.norm1.weight"]
        assert torch.equal(
            dict(u for unit in units for u in unit)["visual.blocks.0.norm1.weight"], torch.full((3,), 2.0)
        )


def test_non_materializing_ranks_yield_nothing_and_read_nothing(tmp_path):
    ckpt = _checkpoint(tmp_path, indexed=True)
    with patch(f"{_MODULE}.safe_open") as opened:
        assert list(checkpoint_towers.iter_checkpoint_tower_units(ckpt, materialize=False)) == []
    opened.assert_not_called()


def test_the_checkpoint_is_read_once_per_path(tmp_path):
    ckpt = _checkpoint(tmp_path, indexed=True)
    checkpoint_towers._CACHE.pop(ckpt, None)
    with patch(f"{_MODULE}.torch.cuda.current_device", return_value="cpu"):
        list(checkpoint_towers.iter_checkpoint_tower_units(ckpt, materialize=True))
        with patch(f"{_MODULE}.safe_open") as opened:
            list(checkpoint_towers.iter_checkpoint_tower_units(ckpt, materialize=True))
    opened.assert_not_called()
