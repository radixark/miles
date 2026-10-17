"""Slice and renumber Step5 decoder layers while preserving shared and vision weights.

Drop unselected decoder and MTP layers, retain tensor dtypes and FP8 scales, and
rewrite all per-layer config lists to match the selected layers.
"""

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
# Text-config lists with one entry per layer (decoder layers, then MTP layers).
_PER_LAYER_LISTS = ("layer_types", "partial_rotary_factors", "rope_theta", "swiglu_limits", "swiglu_limits_shared")
_AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)
_SHARD_BYTES = 5 * 1024**3


@dataclass(frozen=True)
class SlicePlan:
    layer_map: dict[int, int]  # original decoder layer index -> new index
    config: dict


def rename(name: str, layer_map: dict[int, int]) -> str | None:
    """New name for a tensor or module name, or None if its layer is dropped."""
    match = _LAYER.match(name)
    if match is None:
        return name
    new_index = layer_map.get(int(match.group(1)))
    return None if new_index is None else f"model.layers.{new_index}.{match.group(2)}"


def _moe_layers(text_config: dict) -> list[int]:
    listed = text_config.get("moe_layer_list")
    enum = text_config.get("moe_layers_enum")
    from_enum = [int(i) for i in enum.split(",") if i.strip()] if isinstance(enum, str) else None
    if listed is not None and from_enum is not None and [int(i) for i in listed] != from_enum:
        raise ValueError("moe_layer_list and moe_layers_enum describe different layers")
    return [int(i) for i in listed] if listed is not None else (from_enum or [])


def plan_slice(config: dict, layers: list[int]) -> SlicePlan:
    text = config["text_config"]
    num_layers = int(text["num_hidden_layers"])
    if not layers or sorted(set(layers)) != layers:
        raise ValueError(f"layers must be unique and ascending, got {layers}")
    if layers[0] < 0 or layers[-1] >= num_layers:
        raise ValueError(f"layers must be decoder layers in [0, {num_layers}); MTP layers cannot be kept")
    layer_map = {old: new for new, old in enumerate(layers)}

    new_text = dict(text)
    new_text["num_hidden_layers"] = len(layers)
    new_text["num_nextn_predict_layers"] = 0
    for key in _PER_LAYER_LISTS:
        if isinstance(text.get(key), list):
            new_text[key] = [text[key][old] for old in layers]
    moe = [layer_map[i] for i in _moe_layers(text) if i in layer_map]
    if "moe_layer_list" in text:
        new_text["moe_layer_list"] = moe
    if "moe_layers_enum" in text:
        new_text["moe_layers_enum"] = ",".join(str(i) for i in moe)

    new_config = {**config, "text_config": new_text}
    for owner in (new_config, new_text):
        quantization = owner.get("quantization_config")
        if isinstance(quantization, dict) and "modules_to_not_convert" in quantization:
            kept = [rename(name, layer_map) for name in quantization["modules_to_not_convert"]]
            owner["quantization_config"] = {
                **quantization,
                "modules_to_not_convert": [name for name in kept if name is not None],
            }
    return SlicePlan(layer_map=layer_map, config=new_config)


class _ShardWriter:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._weight_map: dict[str, str] = {}
        self._total_bytes = 0
        self._num_shards = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self._pending and self._pending_bytes + nbytes > _SHARD_BYTES:
            self._flush()
        self._pending[name] = tensor.contiguous()
        self._pending_bytes += nbytes
        self._total_bytes += nbytes

    def _flush(self) -> None:
        file_name = f"model-{self._num_shards:05d}.safetensors"
        save_file(self._pending, str(self._directory / file_name), metadata={"format": "pt"})
        self._weight_map.update({name: file_name for name in self._pending})
        self._num_shards += 1
        self._pending, self._pending_bytes = {}, 0

    def close(self) -> None:
        if self._pending:
            self._flush()
        index = {"metadata": {"total_size": self._total_bytes}, "weight_map": dict(sorted(self._weight_map.items()))}
        (self._directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def slice_checkpoint(src: Path, dst: Path, layers: list[int]) -> SlicePlan:
    plan = plan_slice(json.loads((src / "config.json").read_text()), layers)
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    names_by_file: dict[str, list[tuple[str, str]]] = {}
    for name, file_name in weight_map.items():
        new_name = rename(name, plan.layer_map)
        if new_name is not None:
            names_by_file.setdefault(file_name, []).append((name, new_name))

    dst.mkdir(parents=True, exist_ok=False)
    writer = _ShardWriter(dst)
    for file_name in sorted(names_by_file):
        with safe_open(str(src / file_name), framework="pt") as shard:
            for name, new_name in names_by_file[file_name]:
                writer.add(new_name, shard.get_tensor(name))
    writer.close()

    (dst / "config.json").write_text(json.dumps(plan.config, indent=2) + "\n")
    for aux in _AUX_FILES:
        if (src / aux).exists():
            shutil.copy2(src / aux, dst / aux)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Step5 checkpoint with the step5 config.json")
    parser.add_argument("--dst", type=Path, required=True, help="output directory (must not exist)")
    parser.add_argument("--layers", required=True, help="comma-separated original decoder layers, e.g. 0,3,4,88,91")
    args = parser.parse_args()
    plan = slice_checkpoint(src=args.src, dst=args.dst, layers=[int(i) for i in args.layers.split(",")])
    print(f"wrote {args.dst}: layers {plan.layer_map}")


if __name__ == "__main__":
    main()
