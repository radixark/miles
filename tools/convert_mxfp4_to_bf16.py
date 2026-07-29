"""
python tools/convert_mxfp4_to_bf16.py [-h] --model-dir MODEL_DIR --save-dir SAVE_DIR
                                      [--device {cpu,cuda}] [--files FILE [FILE ...]]
                                      [--shard-rank SHARD_RANK] [--num-shards NUM_SHARDS]
                                      [--finalize-only] [--overwrite]

Convert an ``mxfp4-pack-quantized`` compressed-tensors safetensors checkpoint to
BF16, replacing every ``<name>.weight_packed`` / ``<name>.weight_scale`` pair with
a dequantized ``<name>.weight`` and dropping ``quantization_config`` from the
copied config.

options:
  --model-dir MODEL_DIR Directory of the HF safetensors quantized model.
  --save-dir SAVE_DIR   Directory to save the converted BF16 model.
  --device {cpu,cuda}   Device used to dequantize (default: cpu).
  --files FILE [FILE ...]
                        Specific safetensors filenames to convert (relative to
                        model-dir). Convert all if omitted.
  --shard-rank SHARD_RANK, --num-shards NUM_SHARDS
                        Convert only this shard of the file list, so the run can
                        be spread over many processes. Both or neither.
  --finalize-only       Skip conversion; only copy metadata and rebuild the
                        safetensors index. Run once after all shards finish.
  --overwrite           Reconvert files that already exist in save-dir.

Example:
--------
python tools/convert_mxfp4_to_bf16.py --model-dir /Kimi-K3/native --save-dir /Kimi-K3/bf16
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

from safetensors.torch import safe_open, save_file

from miles.utils.mxfp4 import dequantize_mxfp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MXFP4-packed compressed-tensors weights to BF16.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--shard-rank", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def quantization_section(config: dict) -> dict:
    """The config level holding ``quantization_config``: nested under
    ``text_config`` for multimodal checkpoints, top level otherwise."""
    return config.get("text_config", config)


def load_group_size(model_dir: Path) -> int:
    with (model_dir / "config.json").open() as file:
        config = json.load(file)
    quantization_config = quantization_section(config)["quantization_config"]
    assert quantization_config["format"] == "mxfp4-pack-quantized", quantization_config["format"]
    weights_config = quantization_config["config_groups"]["group_0"]["weights"]
    assert weights_config["type"] == "float"
    assert weights_config["num_bits"] == 4
    assert weights_config["scale_dtype"] == "torch.uint8"
    return int(weights_config["group_size"])


def copy_metadata(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in model_dir.iterdir():
        if not source.is_file() or source.suffix == ".safetensors" or source.name.endswith(".index.json"):
            continue
        shutil.copy2(source, output_dir / source.name)

    config_path = output_dir / "config.json"
    with config_path.open() as file:
        config = json.load(file)
    del quantization_section(config)["quantization_config"]
    with config_path.open("w") as file:
        json.dump(config, file, indent=2)


def convert_file(
    source_path: Path,
    output_path: Path,
    group_size: int,
    device: str,
) -> None:
    tensors = {}
    with safe_open(source_path, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        packed_keys = sorted(name for name in keys if name.endswith(".weight_packed"))
        scale_keys = {name for name in keys if name.endswith(".weight_scale")}

        for packed_name in packed_keys:
            prefix = packed_name.removesuffix(".weight_packed")
            scale_name = f"{prefix}.weight_scale"
            assert scale_name in scale_keys, f"Missing scale for {packed_name}"
            scale_keys.remove(scale_name)

            weight_packed = reader.get_tensor(packed_name).to(device)
            weight_scale = reader.get_tensor(scale_name).to(device)
            weight = dequantize_mxfp4(weight_packed, weight_scale, group_size).cpu()
            output_name = f"{prefix}.weight"
            tensors[output_name] = weight

        assert not scale_keys, f"Orphan MXFP4 scales in {source_path.name}: {sorted(scale_keys)}"
        quantized_keys = set(packed_keys) | {
            f"{name.removesuffix('.weight_packed')}.weight_scale" for name in packed_keys
        }
        for name in sorted(keys - quantized_keys):
            tensor = reader.get_tensor(name)
            tensors[name] = tensor

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    save_file(tensors, temporary_path)
    os.replace(temporary_path, output_path)


def build_index(output_dir: Path) -> None:
    dtype_sizes = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I32": 4,
        "I64": 8,
        "U8": 1,
    }
    weight_map = {}
    total_size = 0
    for path in sorted(output_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                assert name not in weight_map, f"Duplicate tensor: {name}"
                weight_map[name] = path.name
                tensor_slice = reader.get_slice(name)
                shape = tensor_slice.get_shape()
                dtype = tensor_slice.get_dtype()
                assert dtype in dtype_sizes, f"Unsupported safetensors dtype: {dtype}"
                total_size += math.prod(shape) * dtype_sizes[dtype]

    with (output_dir / "model.safetensors.index.json").open("w") as file:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, file, indent=2)


def main() -> None:
    args = parse_args()
    assert args.model_dir.is_dir(), args.model_dir
    assert args.save_dir != args.model_dir
    assert (args.shard_rank is None) == (args.num_shards is None)
    if args.num_shards is not None:
        assert args.num_shards > 0
        assert 0 <= args.shard_rank < args.num_shards
        assert args.files is None, "--files cannot be combined with sharded conversion"

    group_size = load_group_size(args.model_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        copy_metadata(args.model_dir, args.save_dir)
        build_index(args.save_dir)
        return

    if args.shard_rank in (None, 0):
        copy_metadata(args.model_dir, args.save_dir)

    filenames = args.files or [path.name for path in sorted(args.model_dir.glob("*.safetensors"))]
    if args.num_shards is not None:
        filenames = filenames[args.shard_rank :: args.num_shards]
    assert filenames, "No safetensors files found"
    for filename in filenames:
        source_path = args.model_dir / filename
        output_path = args.save_dir / filename
        assert source_path.is_file(), source_path
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing {output_path}", flush=True)
            continue
        print(f"Converting {source_path} -> {output_path}", flush=True)
        convert_file(source_path, output_path, group_size, args.device)

    if args.num_shards is None:
        build_index(args.save_dir)


if __name__ == "__main__":
    main()
