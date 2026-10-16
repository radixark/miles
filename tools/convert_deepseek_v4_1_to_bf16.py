"""Convert the official DeepSeek-V4.1-Flash checkpoint (fp8 dense + fp4 experts) to the BF16 HF checkpoint
`--load-from-hf` reads; engram tables stay fp8, MTP/DSpark/vision are dropped, config.json is flattened."""

import json
import os
import shutil
import struct
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

from miles.utils.mxfp4 import dequantize_mxfp4

MXFP4_BLOCK = 32
COPY_CHUNK_BYTES = 256 * 1024**2

_DROPPED_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")
_DROPPED_SUFFIXES = (".ffn.gate.bias_vl",)
_RAW_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")
_DROPPED_CONFIG_KEYS = ("quantization_config", "vision_config", "text_config")
# E8M0 scales are read as their raw exponent bytes.
_TORCH_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "I8": torch.int8}


def read_header(path: Path) -> tuple[dict, int]:
    """Tensor entries of a safetensors file and the byte offset where its data starts."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def read_tensor(path: Path, entry: dict, data_start: int, device: str) -> torch.Tensor:
    begin, end = entry["data_offsets"]
    with path.open("rb") as f:
        f.seek(data_start + begin)
        data = bytearray(f.read(end - begin))
    tensor = torch.frombuffer(data, dtype=torch.uint8)
    return tensor.view(_TORCH_DTYPES[entry["dtype"]]).reshape(entry["shape"]).to(device)


def dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    rows, cols = weight.shape
    assert scale.shape == (-(-rows // block[0]), -(-cols // block[1])), f"{weight.shape=} {scale.shape=}"
    scale = torch.exp2(scale.float() - 127.0)
    scale = scale.repeat_interleave(block[0], dim=0)[:rows].repeat_interleave(block[1], dim=1)[:, :cols]
    return (weight.float() * scale).to(torch.bfloat16)


def is_kept(name: str) -> bool:
    return not name.startswith(_DROPPED_PREFIXES) and not name.endswith(_DROPPED_SUFFIXES)


def is_quantized(name: str, entry: dict) -> bool:
    return entry["dtype"] in ("F8_E4M3", "I8") and not name.endswith(_RAW_SUFFIXES)


def convert_config(config: dict) -> dict:
    """Flatten text_config to the top level, the layout `_normalize_deepseek_v41_kwargs` also produces."""
    text = {k: v for k, v in config["text_config"].items() if k != "model_type" and not k.startswith("dspark_")}
    text["num_nextn_predict_layers"] = 0
    return {**text, **{k: v for k, v in config.items() if k not in _DROPPED_CONFIG_KEYS}}


def write_safetensors(path: Path, tensors: list[tuple[str, dict, torch.Tensor | tuple[Path, int]]]) -> None:
    """Write `(name, entry, source)` tensors in order; a `(file, offset)` source is copied without loading it."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, entry, _ in tensors:
        nbytes = entry["data_offsets"][1] - entry["data_offsets"][0]
        header[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)

    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)
        for name, entry, source in tensors:
            nbytes = entry["data_offsets"][1] - entry["data_offsets"][0]
            if isinstance(source, torch.Tensor):
                data = source.contiguous().cpu().view(torch.uint8).numpy()
                assert data.nbytes == nbytes, f"{name}: {data.nbytes=} {nbytes=}"
                out.write(data.tobytes())
                continue
            src_path, src_offset = source
            with src_path.open("rb") as f:
                f.seek(src_offset)
                remaining = nbytes
                while remaining:
                    chunk = f.read(min(COPY_CHUNK_BYTES, remaining))
                    assert chunk, f"{name}: {src_path} ended {remaining} bytes early"
                    out.write(chunk)
                    remaining -= len(chunk)
    os.replace(pending, path)


def convert_file(src_path: Path, dst_path: Path, block: tuple[int, int], device: str) -> dict[str, int]:
    header, data_start = read_header(src_path)
    counts = {"fp8": 0, "mxfp4": 0, "copied": 0}
    tensors = []
    for name, entry in header.items():
        if not is_kept(name):
            continue
        if name.endswith(".scale") and not name.endswith(_RAW_SUFFIXES):
            weight_name = name.removesuffix("scale") + "weight"
            assert weight_name in header and is_quantized(weight_name, header[weight_name]), f"orphan scale {name}"
            continue
        if not is_quantized(name, entry):
            tensors.append((name, entry, (src_path, data_start + entry["data_offsets"][0])))
            counts["copied"] += 1
            continue
        scale_name = name.removesuffix("weight") + "scale"
        assert scale_name in header, f"{name} has no scale in {src_path.name}"
        weight = read_tensor(src_path, entry, data_start, device)
        scale = read_tensor(src_path, header[scale_name], data_start, device)
        if entry["dtype"] == "F8_E4M3":
            tensor = dequant_fp8_block(weight, scale, block)
            counts["fp8"] += 1
        else:
            tensor = dequantize_mxfp4(weight.view(torch.uint8), scale, MXFP4_BLOCK)
            counts["mxfp4"] += 1
        nbytes = tensor.numel() * tensor.element_size()
        tensors.append((name, {"dtype": "BF16", "shape": list(tensor.shape), "data_offsets": [0, nbytes]}, tensor))
    if tensors:
        write_safetensors(dst_path, tensors)
    return counts


def build_index(save_dir: Path, config: dict) -> dict:
    weight_map, total_size = {}, 0
    for path in sorted(save_dir.glob("*.safetensors")):
        header, _ = read_header(path)
        for name, entry in header.items():
            assert name not in weight_map, f"duplicate tensor {name}"
            weight_map[name] = path.name
            total_size += entry["data_offsets"][1] - entry["data_offsets"][0]
            assert entry["dtype"] in ("BF16", "F32") or name.endswith(_RAW_SUFFIXES), f"{name} is {entry['dtype']}"
    check_output(weight_map, config)
    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def check_output(weight_map: dict[str, str], config: dict) -> None:
    """Every decoder layer has its attention, router and all routed experts; engram layers have their tables."""
    for layer in range(config["num_hidden_layers"]):
        prefix = f"layers.{layer}."
        required = {"attn.wq_a.weight", "attn.wkv.weight", "attn.wo_a.weight", "attn.wo_b.weight", "ffn.gate.weight"}
        required |= {f"ffn.experts.{e}.w{i}.weight" for e in range(config["n_routed_experts"]) for i in (1, 2, 3)}
        if layer in config["engram_layer_ids"]:
            required |= {"engram.embed.weight", "engram.embed.scale", "engram.wkv.weight"}
        missing = {prefix + name for name in required} - weight_map.keys()
        assert not missing, f"layer {layer} is missing {sorted(missing)[:4]} ({len(missing)} total)"
    for name in ("embed.weight", "head.weight", "norm.weight"):
        assert name in weight_map, f"missing {name}"


def finalize(src: Path, dst: Path, config: dict) -> None:
    new_config = convert_config(config)
    index = build_index(dst, new_config)
    for path in src.iterdir():
        if path.is_file() and not path.name.startswith(".") and path.suffix != ".safetensors":
            if path.name not in ("config.json", "model.safetensors.index.json"):
                shutil.copy2(path, dst / path.name)
    (dst / "config.json").write_text(json.dumps(new_config, indent=2) + "\n")
    index_path = dst / "model.safetensors.index.json"
    pending = index_path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(index, indent=2) + "\n")
    pending.replace(index_path)
    print(f"{len(index['weight_map'])} tensors, {index['metadata']['total_size'] / 1e9:.2f} GB -> {dst}")


def main(
    model_dir: str,
    save_dir: str,
    device: str,
    shard_rank: int | None,
    num_shards: int | None,
    finalize_only: bool,
    overwrite: bool,
):
    src, dst = Path(model_dir), Path(save_dir)
    assert src.resolve() != dst.resolve()
    assert (shard_rank is None) == (num_shards is None)
    dst.mkdir(parents=True, exist_ok=True)
    config = json.loads((src / "config.json").read_text())
    assert config["architectures"] == ["DeepseekV41ForCausalLM"], config["architectures"]
    quant = config["quantization_config"]
    assert quant["scale_fmt"] == "ue8m0" and quant["expert_dtype"] == "fp4", quant
    block = tuple(quant["weight_block_size"])

    if not finalize_only:
        file_names = sorted(set(json.loads((src / "model.safetensors.index.json").read_text())["weight_map"].values()))
        if num_shards is not None:
            file_names = file_names[shard_rank::num_shards]
        counts = {"fp8": 0, "mxfp4": 0, "copied": 0}
        for file_name in tqdm(file_names, desc="files"):
            if (dst / file_name).exists() and not overwrite:
                print(f"Skipping existing {dst / file_name}", flush=True)
                continue
            for key, value in convert_file(src / file_name, dst / file_name, block, device).items():
                counts[key] += value
        print(counts, flush=True)
    if finalize_only or num_shards is None:
        finalize(src, dst, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--shard-rank", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(
        args.model_dir,
        args.save_dir,
        args.device,
        args.shard_rank,
        args.num_shards,
        args.finalize_only,
        args.overwrite,
    )
