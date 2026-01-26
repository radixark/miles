#!/usr/bin/env python3
"""Visualize Megatron distributed checkpoint tensor shapes and dtypes."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import Metadata


def load_metadata(ckpt_path: Path) -> Metadata:
    storage_reader = dcp.FileSystemReader(ckpt_path)
    return storage_reader.read_metadata()


def visualize_checkpoint(ckpt_path: str, filter_pattern: str | None = None, sort_by: str = "name"):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    if (ckpt_path / "latest_checkpointed_iteration.txt").exists():
        with open(ckpt_path / "latest_checkpointed_iteration.txt") as f:
            iteration = f.read().strip()
        if iteration == "release":
            ckpt_path = ckpt_path / "release"
        else:
            ckpt_path = ckpt_path / f"iter_{iteration.zfill(7)}"
        print(f"Using checkpoint: {ckpt_path}")

    metadata = load_metadata(ckpt_path)

    print("\n" + "=" * 100)
    print("TENSOR INFORMATION")
    print("=" * 100)

    tensors_info = []
    for fqn, tensor_meta in metadata.state_dict_metadata.items():
        if hasattr(tensor_meta, "size"):
            shape = tuple(tensor_meta.size)
            dtype = tensor_meta.properties.dtype if hasattr(tensor_meta, "properties") else "unknown"
            tensors_info.append({
                "name": fqn,
                "shape": shape,
                "dtype": str(dtype),
                "numel": _prod(shape),
            })

    if filter_pattern:
        tensors_info = [t for t in tensors_info if filter_pattern in t["name"]]

    if sort_by == "name":
        tensors_info.sort(key=lambda x: x["name"])
    elif sort_by == "size":
        tensors_info.sort(key=lambda x: -x["numel"])
    elif sort_by == "dtype":
        tensors_info.sort(key=lambda x: (x["dtype"], x["name"]))

    max_name_len = max((len(t["name"]) for t in tensors_info), default=40)
    max_name_len = min(max_name_len, 100)

    total_params = 0
    total_bytes = 0

    print(f"\n{'Name':<{max_name_len}}  {'Shape':<40}  {'Dtype':<20}  {'Params':>15}")
    print("-" * (max_name_len + 80))

    for info in tensors_info:
        name = info["name"]
        if len(name) > max_name_len:
            name = "..." + name[-(max_name_len - 3):]

        shape_str = str(info["shape"])
        if len(shape_str) > 38:
            shape_str = shape_str[:35] + "..."

        numel = info["numel"]
        total_params += numel
        total_bytes += numel * _dtype_bytes(info["dtype"])

        print(f"{name:<{max_name_len}}  {shape_str:<40}  {info['dtype']:<20}  {numel:>15,}")

    print("-" * (max_name_len + 80))
    print(f"\nTotal tensors: {len(tensors_info)}")
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Estimated size: {total_bytes / 1e9:.2f} GB")

    if (ckpt_path / "metadata.json").exists():
        print("\n" + "=" * 100)
        print("METADATA.JSON")
        print("=" * 100)
        with open(ckpt_path / "metadata.json") as f:
            meta = json.load(f)
        print(json.dumps(meta, indent=2))


def _prod(shape):
    result = 1
    for s in shape:
        result *= s
    return result


def _dtype_bytes(dtype_str: str) -> int:
    dtype_map = {
        "torch.float32": 4,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float8_e4m3fn": 1,
        "torch.float8_e5m2": 1,
        "torch.int8": 1,
        "torch.int16": 2,
        "torch.int32": 4,
        "torch.int64": 8,
        "torch.bool": 1,
    }
    return dtype_map.get(dtype_str, 2)


def main():
    parser = argparse.ArgumentParser(description="Visualize Megatron distributed checkpoint")
    parser.add_argument("ckpt_path", type=str, help="Path to checkpoint directory")
    parser.add_argument("--filter", "-f", type=str, default=None, help="Filter tensors by name pattern")
    parser.add_argument(
        "--sort",
        "-s",
        type=str,
        default="name",
        choices=["name", "size", "dtype"],
        help="Sort tensors by: name, size, or dtype",
    )
    args = parser.parse_args()

    visualize_checkpoint(args.ckpt_path, filter_pattern=args.filter, sort_by=args.sort)


if __name__ == "__main__":
    main()
