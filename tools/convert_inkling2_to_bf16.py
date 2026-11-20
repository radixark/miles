"""
Dequantize an Inkling 2 Mini checkpoint's NVFP4 / MXFP8 weights to BF16, so the torchtitan trainer and a BF16
SGLang engine read the same weights.

NVFP4: value = e2m1(code) * e4m3_block_scale[k // 16] * amax / outer_scale_denominator
MXFP8: value = e4m3 * 2 ** (ue8m0[k // 32] - 127)

`hf_quant_config.json` moves the dequantized tensors into its BF16 group. Every non-weight file (config, tokenizer,
`mtp.safetensors`) is copied unchanged.

python tools/convert_inkling2_to_bf16.py --model-dir <src> --save-dir <dst>
"""

import json
import shutil
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
SCALE_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")


def dequant_nvfp4(packed, scale, amax, denominator):
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    values = E2M1_TABLE.to(packed.device)[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    values = values.view(*values.shape[:-1], -1, NVFP4_BLOCK) * scale.float().unsqueeze(-1)
    values = values.flatten(-2) * (amax.float() / denominator.float()).view(-1, 1, 1)
    return values.to(torch.bfloat16)


def dequant_mxfp8(weight, scale):
    values = weight.float().view(*weight.shape[:-1], -1, MXFP8_BLOCK) * torch.exp2(scale.float() - 127.0).unsqueeze(-1)
    return values.flatten(-2).to(torch.bfloat16)


def _shard_of(key: str) -> str:
    if ".llm.layers." in key:
        return f"layer{int(key.split('.llm.layers.')[1].split('.')[0]):02d}"
    return "visual" if ".visual." in key else "misc"


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    src, dst = args.model_dir, args.save_dir
    dst.mkdir(parents=True, exist_ok=True)

    quant = json.loads((src / "hf_quant_config.json").read_text())
    groups = quant["model"]["parameter_groups"]
    nvfp4 = {t for g in groups if g["compute_dtype"] == "nvfp4" for t in g["tensors"]}
    mxfp8 = {t for g in groups if g["compute_dtype"] == "mxfp8" for t in g["tensors"]}
    bf16_group = next(g for g in groups if g["compute_dtype"] == "bfloat16" and "fwd_output_dtype" not in g)
    bf16_group["tensors"] = sorted(bf16_group["tensors"] + sorted(nvfp4 | mxfp8))
    quant["model"]["parameter_groups"] = [g for g in groups if g["compute_dtype"] not in ("nvfp4", "mxfp8")]

    index = src / "model.safetensors.index.json"
    source_files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else sorted(p.name for p in src.glob("model*.safetensors"))
    )
    scales = {name + suffix for name in nvfp4 | mxfp8 for suffix in SCALE_SUFFIXES}
    shards = defaultdict(dict)
    for filename in source_files:
        with safe_open(src / filename, "pt", device=args.device) as f:
            for key in sorted(f.keys()):
                if key in scales:
                    continue
                if key in nvfp4:
                    tensor = dequant_nvfp4(
                        f.get_tensor(key),
                        f.get_tensor(key + "_scale"),
                        f.get_tensor(key + "_amax"),
                        f.get_tensor(key + "_outer_scale_denominator"),
                    )
                elif key in mxfp8:
                    tensor = dequant_mxfp8(f.get_tensor(key), f.get_tensor(key + "_scale"))
                else:
                    tensor = f.get_tensor(key)
                shards[_shard_of(key)][key] = tensor.cpu().contiguous()

    names = sorted(shards)
    weight_map = {}
    for i, shard in enumerate(names):
        filename = f"model-{i + 1:05d}-of-{len(names):05d}.safetensors"
        save_file(shards[shard], dst / filename, metadata={"format": "pt"})
        weight_map |= {key: filename for key in shards[shard]}
    total = sum(t.numel() * t.element_size() for s in shards.values() for t in s.values())
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    (dst / "hf_quant_config.json").write_text(json.dumps(quant, indent=1))
    for path in src.iterdir():
        if path.name in (*source_files, "model.safetensors.index.json", "hf_quant_config.json", ".cache"):
            continue
        if path.is_dir():
            shutil.copytree(path, dst / path.name, dirs_exist_ok=True)
        else:
            shutil.copy2(path, dst / path.name)
    print(f"wrote {len(names)} shards, {total / 1e9:.1f} GB to {dst}")


if __name__ == "__main__":
    main()
