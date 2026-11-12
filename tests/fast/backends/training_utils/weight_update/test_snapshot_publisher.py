import json
from pathlib import Path
from unittest.mock import patch

import pytest
import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher

_PUBLISHER_MODULE = "miles.backends.training_utils.weight_update.snapshot_publisher"


class _Iterator:
    placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, exported: dict[str, torch.Tensor]) -> None:
        self._exported = exported

    def iter_hf_weights(self, weights):
        yield list(self._exported.items())


def _write_model(base: Path, exported: dict[str, torch.Tensor], output: Path) -> dict[str, torch.Tensor]:
    with patch(f"{_PUBLISHER_MODULE}.dist") as dist:
        dist.get_rank.return_value = 0
        output.mkdir()
        SnapshotPublisher(_Iterator(exported)).write_model(output, weights={}, hf_checkpoint=str(base))
    weight_map = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    return {name: safetensors.torch.load_file(output / shard)[name] for name, shard in weight_map.items()}


def test_the_snapshot_holds_the_base_tensors_the_trainer_does_not_export(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    a, b, c = torch.arange(4.0), torch.ones(2), torch.zeros(3)
    safetensors.torch.save_file({"a": a, "b": b}, base / "model-00001-of-00002.safetensors")
    safetensors.torch.save_file({"c": c}, base / "model-00002-of-00002.safetensors")
    (base / "config.json").write_text("{}")

    snapshot = _write_model(base, {"a": a + 1}, tmp_path / "snapshot")

    assert sorted(snapshot) == ["a", "b", "c"]
    assert torch.equal(snapshot["a"], a + 1)
    assert torch.equal(snapshot["b"], b) and torch.equal(snapshot["c"], c)


def test_the_snapshot_keeps_the_names_of_a_checkpoint_the_engines_rename(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    embed, wq = torch.arange(4.0), torch.ones(2)
    safetensors.torch.save_file({"embed.weight": embed, "layers.0.attn.wq.weight": wq}, base / "model.safetensors")
    (base / "config.json").write_text("{}")
    engine_name_by_base_name = {
        "embed.weight": "model.embed_tokens.weight",
        "layers.0.attn.wq.weight": "model.layers.0.self_attn.wq.weight",
    }

    with patch(f"{_PUBLISHER_MODULE}.get_param_name_remap", return_value=engine_name_by_base_name.__getitem__):
        snapshot = _write_model(base, {"model.layers.0.self_attn.wq.weight": wq + 1}, tmp_path / "snapshot")

    assert sorted(snapshot) == ["embed.weight", "layers.0.attn.wq.weight"]
    assert torch.equal(snapshot["layers.0.attn.wq.weight"], wq + 1)
    assert torch.equal(snapshot["embed.weight"], embed)


def test_an_exported_name_no_checkpoint_tensor_maps_to_is_refused(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    a = torch.ones(2)
    safetensors.torch.save_file({"model.language_model.a": a}, base / "model.safetensors")
    (base / "config.json").write_text("{}")

    with pytest.raises(AssertionError, match=r"\['model.a'\]"):
        _write_model(base, {"model.a": a + 1}, tmp_path / "snapshot")
