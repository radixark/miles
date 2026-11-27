"""A block-FP8 checkpoint as the BF16 reference of TOP's per-layer VJP (miles.utils.debug_utils.top_vjp).

Every E4M3 weight with a 128x128 block scale (DeepSeek-V4's ``.scale``, GLM's ``.weight_scale_inv``) becomes one BF16
tensor; the scale is dropped:

- ``--weights served`` (the reference the VJP judges against): the weight serving computes with at the first rollout.
  Miles syncs the masters before it, so serving holds the sync quantizer's bytes of each master
  (``miles_plugins.top.fp8_masters.served_from_checkpoint``). TOP's backward is evaluated there.
- ``--weights masters``: the BF16 masters the trainer installs (``master_from_checkpoint``: the dequantization moved
  off the weight-sync grid), for the contract-bias report (the gradient at the masters against the one at the served
  weights).

Every other tensor and file is copied as stored, under the checkpoint's own names (for DeepSeek-V4 the names of its
reference, DeepSeek's inference Block: miles.utils.debug_utils.top_reference_dsv4; for GLM HF's).
tools/fp8_cast_bf16.py, the plain dequantization, is neither.

    python tools/convert_fp8_to_bf16_reference.py --weights served --model-dir <FP8 checkpoint> --save-dir <reference>
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles_plugins.top.fp8_masters import master_from_checkpoint, served_from_checkpoint

INDEX = "model.safetensors.index.json"
WEIGHTS = {"served": served_from_checkpoint, "masters": master_from_checkpoint}
SCALE_SUFFIXES = (".weight_scale_inv", ".scale")


def _scaled_weights(weight_map: dict) -> dict[str, str]:
    """{weight name: its block-scale name} for every weight stored with a scale."""
    out = {}
    for name in weight_map:
        for suffix in SCALE_SUFFIXES:
            if name.endswith(suffix) and (weight := f"{name.removesuffix(suffix)}.weight") in weight_map:
                out[weight] = name
                break
    return out


def convert(model_dir: str, save_dir: str, weights: str, device: str) -> None:
    rule = WEIGHTS[weights]
    src, dst = Path(model_dir), Path(save_dir)
    dst.mkdir(parents=True, exist_ok=True)
    weight_map = json.loads((src / INDEX).read_text())["weight_map"]
    scaled = _scaled_weights(weight_map)
    scales = set(scaled.values())

    def read(name):
        with safe_open(src / weight_map[name], framework="pt", device=device) as reader:
            return reader.get_tensor(name)

    for path in src.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != INDEX:
            shutil.copy(path, dst / path.name)
    shards = defaultdict(list)
    for name, shard in weight_map.items():
        if name not in scales:
            shards[shard].append(name)
    for shard, names in sorted(shards.items()):
        tensors = {}
        for name in names:
            tensor = read(name)
            if name in scaled:
                tensor = rule(tensor, read(scaled[name]))
            tensors[name] = tensor.cpu().contiguous()
        save_file(tensors, dst / shard, metadata={"format": "pt"})
        print(f"{shard}: {len(tensors)} tensors", flush=True)
    kept = {name: shard for name, shard in weight_map.items() if name not in scales}
    (dst / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": kept}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--weights", choices=sorted(WEIGHTS), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    convert(args.model_dir, args.save_dir, args.weights, args.device)
