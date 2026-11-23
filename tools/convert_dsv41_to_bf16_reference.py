"""A DeepSeek-V4.1 release checkpoint as the BF16 reference of TOP's per-layer VJP (miles.utils.debug_utils.top_vjp).

Every quantized weight (an E4M3 weight with a 32x32 UE8M0 ``.scale``, ``miles_plugins.top.fp8_b32``; an MXFP4 routed
expert, int8 E2M1 pairs with a 1x32 ``.scale``, ``miles_plugins.top.dsv41_experts``) becomes one BF16 tensor; the
scales are dropped:

- ``--weights served`` (the reference the VJP judges against): the weight serving computes with at the first rollout,
  where TOP's backward is evaluated. Miles syncs the masters before it, and the sync quantizer returns the release's
  own codes for them, so it is the release's exact dequantization (the experts' lossless FP8 cast holds the same
  values).
- ``--weights masters``: the BF16 masters the trainer installs (that dequantization moved off the sync grid), for the
  contract-bias report.

The engram tables are left out: TOP does not train them, and the reference replays the rows TOP looked up
(miles.utils.debug_utils.top_reference_dsv41). Every other tensor and file is copied as stored, under the release's
raw names, which are the names of DeepSeek's inference modules.

    python tools/convert_dsv41_to_bf16_reference.py --weights served --model-dir <release checkpoint> --save-dir <ref>
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles_plugins.top.dsv41_experts import dequantize_mxfp4
from miles_plugins.top.fp8_b32 import dequantize_b32
from miles_plugins.top.off_grid import E4M3_32_UE8M0, MXFP4, offset_off_grid

INDEX = "model.safetensors.index.json"


def served(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.float8_e4m3fn:
        return dequantize_b32(weight, scale)
    if weight.dtype in (torch.int8, torch.uint8):
        return dequantize_mxfp4(weight, scale)
    raise ValueError(f"[top] no V4.1 rule for a {weight.dtype} weight with a scale")


def master(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    grid = E4M3_32_UE8M0 if weight.dtype == torch.float8_e4m3fn else MXFP4
    return offset_off_grid(served(weight, scale), grid)


WEIGHTS = {"served": served, "masters": master}


def convert(model_dir: str, save_dir: str, weights: str, device: str) -> None:
    rule = WEIGHTS[weights]
    src, dst = Path(model_dir), Path(save_dir)
    dst.mkdir(parents=True, exist_ok=True)
    weight_map = json.loads((src / INDEX).read_text())["weight_map"]
    dropped = {name for name in weight_map if ".engram.embed." in name}
    scales = {name for name in weight_map if name.endswith(".scale") and f"{name[:-6]}.weight" in weight_map}

    def read(name):
        with safe_open(src / weight_map[name], framework="pt", device=device) as reader:
            return reader.get_tensor(name)

    for path in src.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and path.name != INDEX:
            shutil.copy(path, dst / path.name)
    shards = defaultdict(list)
    for name, shard in weight_map.items():
        if name not in scales and name not in dropped:
            shards[shard].append(name)
    for shard, names in sorted(shards.items()):
        tensors = {}
        for name in names:
            tensor = read(name)
            if name.endswith(".weight") and (scale := f"{name[:-7]}.scale") in scales:
                tensor = rule(tensor, read(scale))
            tensors[name] = tensor.cpu().contiguous()
        save_file(tensors, dst / shard, metadata={"format": "pt"})
        print(f"{shard}: {len(tensors)} tensors", flush=True)
    kept = {name: shard for name, shard in weight_map.items() if name not in scales and name not in dropped}
    (dst / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": kept}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--weights", choices=sorted(WEIGHTS), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    convert(args.model_dir, args.save_dir, args.weights, args.device)
