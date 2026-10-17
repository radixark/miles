"""Prepare and verify GLM-5.3 image-SFT exports without changing training checkpoints.

Run ``prepare`` after a checkpoint is completely saved, then run Miles' Ray
torch_dist converter on the resulting view. ``finalize`` restores the multimodal
HF namespace, supplements the frozen tower for ``none``, and checks the export
against the original HF index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

_VISUAL_PREFIX = "model.visual."
_HF_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_DCP_LAYER = re.compile(r"^(?:language_model\.)?decoder\.layers\.(\d+)\.")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _read_index(directory: Path) -> dict[str, Any]:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise ValueError(f"Missing weight map in {directory}")
    for filename in index["weight_map"].values():
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Unsafe shard filename: {filename}")
    return index


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _visual_keys(origin: Path) -> set[str]:
    keys = {key for key in _read_index(origin)["weight_map"] if key.startswith(_VISUAL_PREFIX)}
    if not keys:
        raise ValueError(f"No original visual weights in {origin}")
    return keys


def _load_common(checkpoint: Path) -> dict[str, Any]:
    # Only checkpoint preparation needs Megatron's format-aware reader.
    from megatron.core.dist_checkpointing.serialization import load_common_state_dict

    return load_common_state_dict(checkpoint)


def _link_source(source: Path, target: Path) -> None:
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Export view contains a different source: {target}")
    target.symlink_to(source.resolve())


def prepare_checkpoint_view(
    *,
    checkpoint: Path,
    view: Path,
    vision_trainable: str,
    origin_visual: Path,
) -> dict[str, Any]:
    checkpoint, view = checkpoint.resolve(), view.resolve()
    if view == checkpoint or view.is_relative_to(checkpoint):
        raise ValueError("The export view must be outside the training checkpoint")
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    common = _load_common(checkpoint)
    if "args" not in common:
        raise ValueError("The checkpoint does not contain actual training arguments")
    args = common["args"]
    if getattr(args, "glm5_vision_trainable", None) != vision_trainable:
        raise ValueError("Checkpoint visual scope does not match the requested arm")
    model_keys = {
        key
        for key, value in metadata.state_dict_metadata.items()
        if isinstance(value, TensorStorageMetadata) and "optimizer" not in key and "_state" not in key
    }
    layers = {int(match.group(1)) for key in model_keys if (match := _DCP_LAYER.match(key))}
    if layers != set(range(args.num_layers)):
        raise ValueError(f"Incomplete language layer set: {sorted(layers)}")
    visual = {key for key in model_keys if key.startswith("visual.")}
    expected_visual = {key.removeprefix("model.") for key in _visual_keys(origin_visual)}
    if vision_trainable == "none" and visual:
        raise ValueError("Frozen visual weights unexpectedly appear in the training checkpoint")
    if vision_trainable == "full" and visual != expected_visual:
        raise ValueError(f"Incomplete trained visual tower: {len(visual)} of {len(expected_visual)} tensors")

    view.mkdir(parents=True, exist_ok=True)
    for source in checkpoint.iterdir():
        if source.name in {".metadata", "metadata.json"} or source.suffix == ".distcp":
            _link_source(source, view / source.name)
    temporary = view / ".common.pt.tmp"
    torch.save({"args": args}, temporary)
    os.replace(temporary, view / "common.pt")
    report = {
        "checkpoint": str(checkpoint),
        "view": str(view),
        "vision_trainable": vision_trainable,
        "checkpoint_iteration": common.get("iteration"),
        "num_layers": args.num_layers,
        "num_experts": args.num_experts,
        "model_tensors": len(model_keys),
        "visual_tensors": len(visual),
    }
    _write_json(view / "vision_export_source.json", report)
    return report


def _hf_name(name: str) -> str:
    if name.startswith("language_model.model."):
        return "model.language_model." + name.removeprefix("language_model.model.")
    if name.startswith("model.layers."):
        return "model.language_model." + name.removeprefix("model.")
    if name in {"model.embed_tokens.weight", "model.norm.weight"}:
        return "model.language_model." + name.removeprefix("model.")
    return name


def _normalized_weight_map(weight_map: dict[str, str]) -> dict[str, str]:
    normalized = {_hf_name(key): filename for key, filename in weight_map.items()}
    if len(normalized) != len(weight_map):
        raise ValueError("Duplicate tensors after restoring the multimodal namespace")
    return normalized


def _expected_language_keys(origin: Path, num_layers: int) -> set[str]:
    expected = set()
    for name in _read_index(origin)["weight_map"]:
        name = _hf_name(name)
        if name.startswith(_VISUAL_PREFIX):
            continue
        match = _HF_LAYER.match(name)
        if match and int(match.group(1)) >= num_layers:
            continue
        expected.add(name)
    return expected


def _check_language_keys(weight_map: dict[str, str], expected: set[str], num_layers: int) -> None:
    actual = {key for key in weight_map if not key.startswith(_VISUAL_PREFIX)}
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(
            f"Language export differs from the original layout: "
            f"missing={sorted(missing)[:12]}, extra={sorted(extra)[:12]}"
        )
    layers = {int(match.group(1)) for key in actual if (match := _HF_LAYER.match(key))}
    if layers != set(range(num_layers)):
        raise ValueError(f"Export does not contain all {num_layers} language layers")


def _rewrite_shards(output: Path, original_map: dict[str, str]) -> int:
    total_size = 0
    for filename in sorted(set(original_map.values())):
        expected = {key for key, shard in original_map.items() if shard == filename}
        path = output / filename
        with safe_open(path, framework="pt", device="cpu") as source:
            if set(source.keys()) != expected:
                raise ValueError(f"Shard contents disagree with the HF index: {path}")
            rewrite = any(_hf_name(key) != key for key in expected)
            tensors = {}
            for key in source.keys():
                if rewrite:
                    tensor = source.get_tensor(key)
                    tensors[_hf_name(key)] = tensor
                    total_size += tensor.numel() * tensor.element_size()
                else:
                    # Names are already correct; validate the header without paging
                    # the entire multi-hundred-GB export through the shared filesystem.
                    tensor = source.get_slice(key)
                    dtype = tensor.get_dtype()
                    if dtype not in _DTYPE_BYTES:
                        raise ValueError(f"Unsupported safetensors dtype {dtype}: {key}")
                    total_size += math.prod(tensor.get_shape()) * _DTYPE_BYTES[dtype]
            metadata = source.metadata()
        if rewrite:
            temporary = path.with_name(f".{path.name}.tmp")
            save_file(tensors, temporary, metadata=metadata)
            os.replace(temporary, path)
        del tensors
    return total_size


def _supplement_frozen_visual(output: Path, origin: Path, weight_map: dict[str, str], expected: set[str]) -> int:
    origin_map = _read_index(origin)["weight_map"]
    filename = "frozen-visual.safetensors"
    path = output / filename
    if path.exists():
        raise FileExistsError(f"Unindexed frozen visual shard already exists: {path}")
    tensors = {}
    for shard in sorted({origin_map[key] for key in expected}):
        with safe_open(origin / shard, framework="pt", device="cpu") as source:
            for key in sorted(expected):
                if origin_map[key] == shard:
                    tensors[key] = source.get_tensor(key)
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, path)
    weight_map.update(dict.fromkeys(expected, filename))
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _check_visual_values(
    *, output: Path, origin: Path, weight_map: dict[str, str], vision_trainable: str
) -> dict[str, Any]:
    origin_map = _read_index(origin)["weight_map"]
    keys = sorted(key for key in origin_map if key.startswith(_VISUAL_PREFIX))
    changed = 0
    max_abs_update = 0.0
    for key in keys:
        with safe_open(origin / origin_map[key], framework="pt", device="cpu") as source:
            initial = source.get_tensor(key)
        with safe_open(output / weight_map[key], framework="pt", device="cpu") as source:
            trained = source.get_tensor(key)
        if initial.shape != trained.shape or initial.dtype != trained.dtype:
            raise ValueError(f"Visual tensor layout changed: {key}")
        if not torch.isfinite(trained).all().item():
            raise ValueError(f"Nonfinite visual weights: {key}")
        equal = torch.equal(initial, trained)
        changed += not equal
        if not equal:
            max_abs_update = max(max_abs_update, (trained.float() - initial.float()).abs().max().item())
    if vision_trainable == "none" and changed:
        raise ValueError(f"Frozen visual tower changed in {changed} tensors")
    if vision_trainable == "full" and not changed:
        raise ValueError("The full arm has no cumulative visual weight updates")
    return {"visual_tensors": len(keys), "changed_visual_tensors": changed, "max_abs_visual_update": max_abs_update}


def finalize_hf_export(
    *,
    output: Path,
    origin: Path,
    origin_visual: Path,
    vision_trainable: str,
    num_layers: int,
) -> dict[str, Any]:
    index = _read_index(output)
    original_map = index["weight_map"]
    weight_map = _normalized_weight_map(original_map)
    _check_language_keys(weight_map, _expected_language_keys(origin, num_layers), num_layers)
    expected_visual = _visual_keys(origin_visual)
    actual_visual = {key for key in weight_map if key.startswith(_VISUAL_PREFIX)}
    if actual_visual and actual_visual != expected_visual:
        raise ValueError("The export contains an incomplete visual tower")
    if vision_trainable == "full" and actual_visual != expected_visual:
        raise ValueError("The full export must contain trained visual weights")
    total_size = _rewrite_shards(output, original_map)
    if vision_trainable == "none" and not actual_visual:
        total_size += _supplement_frozen_visual(output, origin_visual, weight_map, expected_visual)
    index["weight_map"] = weight_map
    index.setdefault("metadata", {})["total_size"] = total_size
    _write_json(output / "model.safetensors.index.json", index)
    visual_report = _check_visual_values(
        output=output, origin=origin_visual, weight_map=weight_map, vision_trainable=vision_trainable
    )
    report = {
        "output": str(output),
        "vision_trainable": vision_trainable,
        "num_layers": num_layers,
        "language_tensors": len(weight_map) - len(expected_visual),
        "total_size": total_size,
        **visual_report,
    }
    _write_json(output / "vision_export_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--checkpoint", required=True, type=Path)
    prepare.add_argument("--view", required=True, type=Path)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--output", required=True, type=Path)
    finalize.add_argument("--origin", required=True, type=Path)
    finalize.add_argument("--num-layers", type=int, default=45)
    for command in (prepare, finalize):
        command.add_argument("--origin-visual", required=True, type=Path)
        command.add_argument("--vision-trainable", required=True, choices=("none", "full"))
    args = parser.parse_args()
    values = vars(args).copy()
    mode = values.pop("mode")
    report = prepare_checkpoint_view(**values) if mode == "prepare" else finalize_hf_export(**values)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
