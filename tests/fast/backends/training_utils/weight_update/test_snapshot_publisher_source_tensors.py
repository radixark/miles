import json
from pathlib import Path

import pytest
import safetensors.torch
import torch
from tests.ci.ci_register import register_cpu_ci

from miles.backends.training_utils.weight_update.snapshot_publisher import _copy_source_tensors, _is_hf_metadata_file

register_cpu_ci(est_time=5, suite="stage-a-cpu", labels=[])


def _write_indexed_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, path / shard_name)
    index = {
        "metadata": {},
        "weight_map": {name: shard_name for name in tensors},
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))


def test_copy_source_tensors_fills_only_missing_weights(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    _write_indexed_checkpoint(
        source,
        {
            "model.visual.proj.weight": torch.arange(4, dtype=torch.float32),
            "model.visual.norm.weight": torch.ones(2),
            "model.language.weight": torch.zeros(3),
        },
    )
    weight_map = {
        "model.visual.proj.weight": "model-00001.safetensors",
        "model.language.weight": "model-00001.safetensors",
    }

    copied_size = _copy_source_tensors(source, output, weight_map, ["model.visual."])

    assert copied_size == 8
    assert weight_map["model.visual.proj.weight"] == "model-00001.safetensors"
    source_shard = weight_map["model.visual.norm.weight"]
    source_tensors = safetensors.torch.load_file(output / source_shard)
    assert torch.equal(source_tensors["model.visual.norm.weight"], torch.ones(2))
    assert "model.language.weight" not in source_tensors


def test_copy_source_tensors_supports_single_file_checkpoint(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    safetensors.torch.save_file(
        {"model.visual.weight": torch.arange(3)},
        source / "model.safetensors",
    )
    weight_map = {}

    _copy_source_tensors(source, output, weight_map, ["model.visual."])

    assert set(weight_map) == {"model.visual.weight"}


def test_copy_source_tensors_rejects_unmatched_prefix(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    _write_indexed_checkpoint(source, {"model.language.weight": torch.ones(1)})

    with pytest.raises(ValueError, match="no matching weights"):
        _copy_source_tensors(source, output, {}, ["model.visual."])


def test_completion_marker_is_not_metadata(tmp_path):
    marker = tmp_path / ".complete"
    marker.touch()

    assert not _is_hf_metadata_file(marker)
