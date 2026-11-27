"""Merge a dense Qwen3.5 Tinker training checkpoint for SGLang NEXTN evaluation."""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.training_utils.checkpoint.adapter.adapter_state import read_adapter_state_weights
from miles.backends.training_utils.checkpoint.adapter.key_mapping import fused_components, split_fused_lora_b
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    base_model_fingerprint,
    read_parametrization,
)
from miles.utils.hf_utils.weight_mapping import get_checkpoint_weight_map
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets
from miles.utils.lora.utils import matches_lora_target


def _find_targets(weight_map, modules, layout):
    selected = set(modules)
    mapped = {}
    covered = set()
    for key in weight_map:
        module = key.removesuffix(".weight")
        parts = (module,) if module in selected else fused_components(module, layout) or ()
        if matching := selected.intersection(parts):
            mapped[key] = parts
            covered.update(matching)
    if missing := selected - covered:
        raise ValueError(f"base checkpoint is missing adapter targets: {sorted(missing)}")
    return mapped


def _merge_weight(tensor, key, parts, weights, shared_sources, scale, layout):
    slices = [tensor] if len(parts) == 1 else split_fused_lora_b(key.removesuffix(".weight"), tensor, layout)
    for module, destination in zip(parts, slices, strict=True):
        if (b := weights.get(f"{module}.lora_B")) is not None:
            a = weights[f"{shared_sources.get(module, module)}.lora_A"]
            destination.copy_(destination.float().addmm(b, a, alpha=scale))
    return tensor


def merge_mtp_checkpoint(base: Path, checkpoint: Path, output: Path):
    config = json.loads((base / "config.json").read_text())
    layout = get_hf_lora_targets(config)
    adapter = read_parametrization(checkpoint)
    assert layout.mtp and any(
        matches_lora_target(module, target) for module in adapter.target_modules for target in layout.mtp
    ), "expected a Qwen3.5 dense MTP training checkpoint"
    assert adapter.base_model_fingerprint == base_model_fingerprint(config), "adapter and base architecture differ"
    weights = read_adapter_state_weights(checkpoint)
    shared_sources = {module: group[0] for group in adapter.shared_lora_a_groups for module in group}
    scale = adapter.lora_alpha / adapter.lora_rank
    weight_map = get_checkpoint_weight_map(base)
    mapped = _find_targets(weight_map, adapter.target_modules, layout)

    output.mkdir()
    try:
        _write_checkpoint(base, output, weight_map, mapped, weights, shared_sources, scale, layout)
    except BaseException:
        shutil.rmtree(output)
        raise


def _write_checkpoint(base, output, weight_map, mapped, weights, shared_sources, scale, layout):
    shards = set(weight_map.values())
    updated_shards = {weight_map[key] for key in mapped}
    for path in base.iterdir():
        if path.is_file() and path.name not in shards:
            shutil.copy2(path, output / path.name)
    for shard in sorted(shards):
        if shard not in updated_shards:
            shutil.copy2(base / shard, output / shard)
            continue
        with safe_open(base / shard, framework="pt", device="cpu") as source:
            tensors = {}
            for key in source.keys():
                tensor = source.get_tensor(key)
                if key in mapped:
                    tensor = _merge_weight(tensor.clone(), key, mapped[key], weights, shared_sources, scale, layout)
                tensors[key] = tensor
            save_file(tensors, output / shard, metadata=source.metadata())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("base", "checkpoint", "output"):
        parser.add_argument(path, type=Path)
    args = parser.parse_args()
    merge_mtp_checkpoint(args.base, args.checkpoint, args.output)
