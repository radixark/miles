import json
from pathlib import Path
from unittest.mock import patch

import safetensors.torch
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher


class _Iterator:
    placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, exported: dict[str, torch.Tensor]) -> None:
        self._exported = exported

    def iter_hf_weights(self, weights):
        yield list(self._exported.items())


def _write_model(base: Path, exported: dict[str, torch.Tensor], output: Path) -> dict[str, torch.Tensor]:
    with patch("miles.backends.training_utils.weight_update.snapshot_publisher.dist") as dist:
        dist.get_rank.return_value = 0
        output.mkdir()
        SnapshotPublisher(_Iterator(exported)).write_model(output, weights={}, hf_checkpoint=str(base))
    weight_map = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    return {name: safetensors.torch.load_file(output / shard)[name] for name, shard in weight_map.items()}


def test_the_snapshot_holds_the_base_tensors_the_trainer_does_not_export(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    a, b, c = (
        torch.arange(4, dtype=torch.bfloat16),
        torch.ones(2, dtype=torch.float32),
        torch.zeros(3, dtype=torch.uint8),
    )
    first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    safetensors.torch.save_file({"a": a, "b": b}, base / first)
    safetensors.torch.save_file({"c": c}, base / second)
    safetensors.torch.save_file({"x": torch.ones(1)}, base / "consolidated.safetensors")
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": first, "b": first, "c": second}})
    )
    (base / "config.json").write_text("{}")

    snapshot = _write_model(base, {"a": a + 1}, tmp_path / "snapshot")

    assert sorted(snapshot) == ["a", "b", "c"]
    assert torch.equal(snapshot["a"], a + 1)
    assert torch.equal(snapshot["b"], b) and torch.equal(snapshot["c"], c)
    assert (tmp_path / "snapshot" / "config.json").read_text() == "{}"


def test_a_base_without_an_index_is_read_from_every_safetensors_file(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    a, b = torch.arange(4, dtype=torch.bfloat16), torch.ones(2, dtype=torch.float32)
    safetensors.torch.save_file({"a": a, "b": b}, base / "model.safetensors")

    snapshot = _write_model(base, {"a": a + 1}, tmp_path / "snapshot")

    assert sorted(snapshot) == ["a", "b"]
    assert torch.equal(snapshot["a"], a + 1)
    assert torch.equal(snapshot["b"], b)
