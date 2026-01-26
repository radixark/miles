#!/usr/bin/env python3
"""Visualize Megatron distributed checkpoint tensor shapes and dtypes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import Metadata


def load_metadata(ckpt_path: Path) -> Metadata:
    storage_reader = dcp.FileSystemReader(ckpt_path)
    return storage_reader.read_metadata()


def visualize_checkpoint(
    ckpt_path: str,
    filter_pattern: str | None = None,
    sort_by: str = "name",
    show_shards: bool = False,
):
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

    print("\n" + "=" * 120)
    print("TENSOR INFORMATION")
    print("=" * 120)

    tensors_info = []
    for fqn, tensor_meta in metadata.state_dict_metadata.items():
        if hasattr(tensor_meta, "size"):
            shape = tuple(tensor_meta.size)
            dtype = tensor_meta.properties.dtype if hasattr(tensor_meta, "properties") else "unknown"

            chunks_info = []
            if hasattr(tensor_meta, "chunks"):
                for chunk in tensor_meta.chunks:
                    chunk_info = {
                        "offsets": tuple(chunk.offsets),
                        "sizes": tuple(chunk.sizes),
                    }
                    chunks_info.append(chunk_info)

            tensors_info.append({
                "name": fqn,
                "shape": shape,
                "dtype": str(dtype),
                "numel": _prod(shape),
                "num_shards": len(chunks_info),
                "shards": chunks_info,
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
    max_name_len = min(max_name_len, 80)

    total_params = 0
    total_bytes = 0

    print(
        f"\n{'Name':<{max_name_len}}  {'Shape':<35}  {'Dtype':<20}  {'Shards':>6}  {'Params':>15}"
    )
    print("-" * (max_name_len + 85))

    for info in tensors_info:
        name = info["name"]
        if len(name) > max_name_len:
            name = "..." + name[-(max_name_len - 3) :]

        shape_str = str(info["shape"])
        if len(shape_str) > 33:
            shape_str = shape_str[:30] + "..."

        numel = info["numel"]
        total_params += numel
        total_bytes += numel * _dtype_bytes(info["dtype"])

        print(
            f"{name:<{max_name_len}}  {shape_str:<35}  {info['dtype']:<20}  {info['num_shards']:>6}  {numel:>15,}"
        )

        if show_shards and info["shards"]:
            for i, shard in enumerate(info["shards"]):
                print(f"    └─ shard[{i}]: offsets={shard['offsets']}, sizes={shard['sizes']}")

    print("-" * (max_name_len + 85))
    print(f"\nTotal tensors: {len(tensors_info)}")
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Estimated size: {total_bytes / 1e9:.2f} GB")

    _print_shard_distribution(metadata)

    if (ckpt_path / "metadata.json").exists():
        print("\n" + "=" * 120)
        print("METADATA.JSON")
        print("=" * 120)
        with open(ckpt_path / "metadata.json") as f:
            meta = json.load(f)
        print(json.dumps(meta, indent=2))


def _print_shard_distribution(metadata: Metadata):
    print("\n" + "=" * 120)
    print("SHARD FILE DISTRIBUTION")
    print("=" * 120)

    file_to_tensors = defaultdict(list)
    file_sizes = defaultdict(int)

    for fqn, tensor_meta in metadata.state_dict_metadata.items():
        if hasattr(tensor_meta, "chunks"):
            dtype = tensor_meta.properties.dtype if hasattr(tensor_meta, "properties") else None
            dtype_bytes = _dtype_bytes(str(dtype)) if dtype else 2

            for chunk in tensor_meta.chunks:
                if hasattr(chunk, "storage_metadata"):
                    storage = chunk.storage_metadata
                    if hasattr(storage, "filename"):
                        filename = storage.filename
                    elif hasattr(storage, "relative_path"):
                        filename = storage.relative_path
                    else:
                        filename = str(storage)
                else:
                    filename = "unknown"

                chunk_size = _prod(chunk.sizes) * dtype_bytes
                file_to_tensors[filename].append({
                    "name": fqn,
                    "offsets": tuple(chunk.offsets),
                    "sizes": tuple(chunk.sizes),
                })
                file_sizes[filename] += chunk_size

    print(f"\n{'File':<30}  {'Tensors':>10}  {'Size (GB)':>12}")
    print("-" * 55)

    for filename in sorted(file_to_tensors.keys()):
        num_tensors = len(file_to_tensors[filename])
        size_gb = file_sizes[filename] / 1e9
        print(f"{filename:<30}  {num_tensors:>10}  {size_gb:>12.2f}")

    print("-" * 55)
    print(f"{'Total':<30}  {sum(len(v) for v in file_to_tensors.values()):>10}  {sum(file_sizes.values()) / 1e9:>12.2f}")

    print("\n" + "=" * 120)
    print("SHARD DETAILS BY FILE")
    print("=" * 120)

    for filename in sorted(file_to_tensors.keys()):
        print(f"\n[{filename}] ({len(file_to_tensors[filename])} chunks)")
        for item in file_to_tensors[filename][:20]:
            name = item["name"]
            if len(name) > 60:
                name = "..." + name[-57:]
            print(f"  {name:<60}  offsets={item['offsets']}, sizes={item['sizes']}")
        if len(file_to_tensors[filename]) > 20:
            print(f"  ... and {len(file_to_tensors[filename]) - 20} more chunks")


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
    parser.add_argument(
        "--shards",
        action="store_true",
        help="Show detailed shard info for each tensor",
    )
    args = parser.parse_args()

    visualize_checkpoint(args.ckpt_path, filter_pattern=args.filter, sort_by=args.sort, show_shards=args.shards)


if __name__ == "__main__":
    main()
