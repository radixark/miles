"""Carve a last-N-layer Muse Spark partial checkpoint out of the released HF checkpoint.

Takes the last ``--layers`` decoder layers (``model.language_model.layers.N.*``) and renumbers them from 0,
keeps every other tensor (embeddings, final norm, lm_head and the vision/audio towers, so the omni engine
loads it natively), and truncates the per-layer lists of ``text_config`` (``layer_types``,
``layer_rope_theta``, ``mlp_layer_types``) to the same layers. Taking the *last* layers keeps the NoPE /
sliding-window schedule, which counts from the top: layer 61 is full attention and so is new layer 3. Every
other top-level file (tokenizer, chat template, processor config) is copied. The text tensor set is checked
against the plugin's ``hf_tensor_specs`` so the Miles loader accepts it as-is.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from miles_plugins.models.muse_spark.config import parse_muse_spark_1x_config
from miles_plugins.models.muse_spark.weight_mapping import NON_TEXT_PREFIXES, hf_tensor_specs

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.*)$")
INDEX_FILE = "model.safetensors.index.json"
SHARD_BYTES = 8 * 2**30


def _reduce_config(config: dict, keep: list[int]) -> tuple[dict, list[str]]:
    """Return the config of the kept layers and the per-layer ``text_config`` keys that were truncated."""
    reduced = copy.deepcopy(config)
    text = reduced["text_config"]
    total = text["num_hidden_layers"]
    per_layer = sorted(k for k, v in text.items() if isinstance(v, list) and len(v) == total)
    for key in per_layer:
        text[key] = [text[key][i] for i in keep]
    text["num_hidden_layers"] = len(keep)
    return reduced, per_layer


def _plan_tensors(weight_map: dict[str, str], keep: list[int]) -> tuple[dict[str, tuple[str, str]], int]:
    """Map each output tensor name to (source name, source shard); count the dropped layer tensors."""
    plan, dropped = {}, 0
    for name, shard in weight_map.items():
        match = LAYER_RE.match(name)
        if match is None:
            plan[name] = (name, shard)
        elif int(match.group(1)) in keep:
            plan[f"model.language_model.layers.{keep.index(int(match.group(1)))}.{match.group(2)}"] = (name, shard)
        else:
            dropped += 1
    return plan, dropped


def _write_shards(src: Path, dst: Path, plan: dict[str, tuple[str, str]], expected: dict) -> tuple[dict, dict]:
    """Copy the planned tensors into ~8 GiB shards; return the output weight map and the tensor inventory."""
    by_shard = defaultdict(list)
    for new_name, (old_name, shard) in plan.items():
        by_shard[shard].append((old_name, new_name))
    out_index, inventory, current, current_bytes = {}, {}, {}, 0

    def flush():
        nonlocal current, current_bytes
        if current:
            fname = f"model-{len(set(out_index.values())):05d}.safetensors"
            save_file(current, str(dst / fname), metadata={"format": "pt"})
            out_index.update(dict.fromkeys(current, fname))
            current, current_bytes = {}, 0

    for shard, pairs in sorted(by_shard.items()):
        with safe_open(src / shard, framework="pt", device="cpu") as reader:
            for old_name, new_name in sorted(pairs):
                tensor = reader.get_tensor(old_name).contiguous()
                spec = expected.get(new_name)
                if spec is not None and (tuple(tensor.shape) != tuple(spec.shape) or tensor.dtype != spec.dtype):
                    raise SystemExit(f"{new_name}: checkpoint {tuple(tensor.shape)} {tensor.dtype} vs expected {spec}")
                inventory[new_name] = {"from": old_name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                nbytes = tensor.numel() * tensor.element_size()
                if current_bytes + nbytes > SHARD_BYTES:
                    flush()
                current[new_name] = tensor
                current_bytes += nbytes
    flush()
    return out_index, inventory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    src, dst, evidence = Path(args.src), Path(args.dst), Path(args.evidence)
    dst.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    started = time.time()

    config = json.loads((src / "config.json").read_text())
    total = config["text_config"]["num_hidden_layers"]
    keep = list(range(total - args.layers, total))
    new_config, per_layer_keys = _reduce_config(config, keep)
    plan, dropped = _plan_tensors(json.loads((src / INDEX_FILE).read_text())["weight_map"], keep)

    # loader contract: the text names must be exactly what the plugin expects for the reduced config
    expected = hf_tensor_specs(parse_muse_spark_1x_config(new_config))
    text_names = {name for name in plan if not name.startswith(NON_TEXT_PREFIXES)}
    if text_names != set(expected):
        raise SystemExit(
            f"text tensor set mismatch: missing={sorted(set(expected) - text_names)[:5]} "
            f"unexpected={sorted(text_names - set(expected))[:5]}"
        )

    out_index, inventory = _write_shards(src, dst, plan, expected)
    total_bytes = sum((dst / fname).stat().st_size for fname in set(out_index.values()))
    (dst / INDEX_FILE).write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": out_index}, indent=2)
    )
    (dst / "config.json").write_text(json.dumps(new_config, indent=2))
    copied = []
    for path in sorted(src.iterdir()):
        if path.is_file() and not path.name.startswith(".") and path.suffix != ".safetensors":
            if path.name not in (INDEX_FILE, "config.json"):
                shutil.copy2(path, dst / path.name)
                copied.append(path.name)
    for fname in sorted(set(out_index.values())):
        inventory[f"__sha256__/{fname}"] = _sha256(dst / fname)
    summary = {
        "src": str(src),
        "dst": str(dst),
        "kept_layers_original": keep,
        "truncated_per_layer_keys": per_layer_keys,
        "tensors": len(out_index),
        "dropped_layer_tensors": dropped,
        "shards": len(set(out_index.values())),
        "gib": round(total_bytes / 2**30, 2),
        "copied_files": copied,
        "seconds": round(time.time() - started, 1),
    }
    (evidence / "partial-checkpoint-inventory.json").write_text(
        json.dumps({"summary": summary, "tensors": inventory}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
