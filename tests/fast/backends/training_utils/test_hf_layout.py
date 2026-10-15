"""An HF export written into the base checkpoint's shard layout: unchanged weights reproduce the base files byte for
byte with any number of writer ranks, trained slices land in place, and an incomplete export fails."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from miles.backends.training_utils.checkpoint.hf_layout import CheckpointLayout, CheckpointLayoutWriter

_SHARDS = {
    "model-00001-of-00002.safetensors": ("experts.w", "tower.w"),
    "model-00002-of-00002.safetensors": ("embed", "router"),
}
_EXPERT_BYTES = 2 * 3 * 2  # one [2, 3] bf16 expert of experts.w


def _locate(name):
    if name.startswith("experts."):
        _, expert, leaf = name.split(".")
        return f"experts.{leaf}", int(expert) * _EXPERT_BYTES
    return name, 0


_LAYOUT = CheckpointLayout(locate=_locate, frozen_prefixes=("tower.",))


def _write_base(path):
    tensors = {
        "experts.w": (torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) / 7).bfloat16(),
        "tower.w": torch.arange(4, dtype=torch.float32).bfloat16(),
        "embed": (torch.arange(15, dtype=torch.float32).reshape(5, 3) / 3).bfloat16(),
        "router": torch.arange(6, dtype=torch.float32).reshape(2, 3) / 9,
    }
    path.mkdir()
    weight_map = {}
    for shard, names in _SHARDS.items():
        save_file({name: tensors[name] for name in names}, path / shard, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(names, shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tensors


def _stream(tensors):
    """The export as the trainer streams it: one tensor per expert, and the frozen tower re-sent with new values."""
    named = [(f"experts.{expert}.w", tensors["experts.w"][expert]) for expert in range(4)]
    return named + [("embed", tensors["embed"]), ("router", tensors["router"]), ("tower.w", tensors["tower.w"] + 1)]


def _export(base, out, stream, world_size):
    out.mkdir()
    for rank in range(world_size):
        writer = CheckpointLayoutWriter(out, base, _LAYOUT, rank=rank, world_size=world_size)
        for name, tensor in stream:
            writer.write(name, tensor)
        writer.finish()


@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_unchanged_weights_export_the_base_files_byte_for_byte(tmp_path, world_size):
    tensors = _write_base(tmp_path / "base")
    _export(tmp_path / "base", tmp_path / "out", _stream(tensors), world_size)
    for shard in _SHARDS:
        assert (tmp_path / "out" / shard).read_bytes() == (tmp_path / "base" / shard).read_bytes()


def test_trained_slices_land_in_place_and_frozen_tensors_stay_base(tmp_path):
    tensors = _write_base(tmp_path / "base")
    trained = dict(tensors, router=tensors["router"] * 2)
    trained["experts.w"] = tensors["experts.w"].clone()
    trained["experts.w"][2] += 1
    _export(tmp_path / "base", tmp_path / "out", _stream(trained), world_size=2)
    exported = {}
    for shard in _SHARDS:
        exported.update(load_file(tmp_path / "out" / shard))
    assert torch.equal(exported["experts.w"], trained["experts.w"])
    assert torch.equal(exported["router"], trained["router"])
    assert torch.equal(exported["embed"], tensors["embed"])
    assert torch.equal(exported["tower.w"], tensors["tower.w"])


@pytest.mark.parametrize(
    "problem, error",
    [("missing", "gap"), ("duplicate", "overlap"), ("unknown", "lacks"), ("dtype", "dtype"), ("shape", "shape")],
)
def test_incomplete_or_mismatched_export_fails(tmp_path, problem, error):
    tensors = _write_base(tmp_path / "base")
    stream = _stream(tensors)
    if problem == "missing":
        stream = [item for item in stream if item[0] != "experts.1.w"]
    elif problem == "duplicate":
        stream = stream + [("experts.1.w", tensors["experts.w"][1])]
    elif problem == "unknown":
        stream = stream + [("lm_head", tensors["embed"])]
    elif problem == "dtype":
        stream = [(name, tensor.float() if name == "embed" else tensor) for name, tensor in stream]
    else:
        stream = [(name, tensor.reshape(3, 5) if name == "embed" else tensor) for name, tensor in stream]
    with pytest.raises((KeyError, ValueError), match=error):
        _export(tmp_path / "base", tmp_path / "out", stream, world_size=1)
