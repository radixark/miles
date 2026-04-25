"""
Dequantize DeepSeek-V4-Flash (MXFP4 experts + block-FP8 attn) into a BF16 HF
checkpoint that fp8_cast_bf16.py / mbridge can consume.

Usage:
  python tools/dsv4_flash_to_bf16.py \
      --input-flash-hf-path /cluster_public/miles_data/models/DeepSeek-V4-Flash-4layer \
      --output-bf16-hf-path /cluster_public/miles_data/models/DeepSeek-V4-285B-4layer-bf16
"""

import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import load_file, save_file
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from tqdm import tqdm


FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)


def dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8-packed FP4 (E2M1) + per-32 E8M0 scale → bfloat16.

    weight: int8 [out, in_bytes]   (each byte holds 2 FP4 elements)
    scale:  e8m0  [out, in/32]
    returns: bf16 [out, in_logical = in_bytes * 2]
    """
    assert weight.dtype == torch.int8
    assert scale.dtype == torch.float8_e8m0fnu
    out_dim, in_bytes = weight.shape
    in_logical = in_bytes * 2
    fp4_block = 32
    assert in_logical % fp4_block == 0
    assert scale.shape == (out_dim, in_logical // fp4_block)

    table = FP4_TABLE.to(weight.device)
    u = weight.view(torch.uint8)
    low = (u & 0x0F).long()
    high = ((u >> 4) & 0x0F).long()
    # interleave low/high to recover logical layout: [..., 2*c]=low, [..., 2*c+1]=high
    decoded = torch.stack([table[low], table[high]], dim=-1).reshape(out_dim, in_logical)

    s = scale.float()  # [out, in_logical/32]
    s = s.repeat_interleave(fp4_block, dim=1)  # [out, in_logical]
    return (decoded * s).to(torch.bfloat16)


def dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor, block: int = 128) -> torch.Tensor:
    """FP8 (E4M3) + per-(block, block) E8M0 scale → bfloat16.

    weight: fp8_e4m3fn [out, in]
    scale:  e8m0       [out/block, in/block]
    """
    assert weight.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float8_e8m0fnu
    out_dim, in_dim = weight.shape
    assert out_dim % block == 0 and in_dim % block == 0
    assert scale.shape == (out_dim // block, in_dim // block)

    s = scale.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    w = weight.float()
    return (w * s).to(torch.bfloat16)


def main(flash_path: str, bf16_path: str) -> None:
    os.makedirs(bf16_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index_file = os.path.join(flash_path, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)
    raw_weight_map: dict[str, str] = index["weight_map"]

    # Remap weight names to HF/DPSK format using sglang helper (handles MTP layers too).
    num_hidden_layers = json.load(open(os.path.join(flash_path, "config.json"))).get("num_hidden_layers")
    remap = lambda name: DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(
        name, is_nextn=name.startswith("mtp."), num_hidden_layers=num_hidden_layers
    )

    # Pre-load ALL scales globally so a weight in shard N can find its scale even
    # if the scale lives in a different shard.
    scale_keys_global = [k for k in raw_weight_map if k.endswith(".scale")]
    scales_by_shard: dict[str, dict[str, torch.Tensor]] = {}
    for sk in scale_keys_global:
        scales_by_shard.setdefault(raw_weight_map[sk], {})[sk] = None
    for shard_name, sk_map in tqdm(scales_by_shard.items(), desc="load scales"):
        shard_dict = load_file(os.path.join(flash_path, shard_name), device=device)
        for k in sk_map:
            sk_map[k] = shard_dict[k]
        del shard_dict
    all_scales: dict[str, torch.Tensor] = {}
    for sk_map in scales_by_shard.values():
        all_scales.update(sk_map)

    new_weight_map: dict[str, str] = {}
    safetensor_files = sorted(glob(os.path.join(flash_path, "*.safetensors")))

    for shard_path in tqdm(safetensor_files, desc="shards"):
        shard_name = os.path.basename(shard_path)
        raw = load_file(shard_path, device=device)

        raw_keys = list(raw.keys())
        scale_keys = {k for k in raw_keys if k.endswith(".scale")}
        weight_keys = {k for k in raw_keys if k.endswith(".weight")}

        new_state: dict[str, torch.Tensor] = {}
        for k in raw_keys:
            if k in scale_keys:
                continue  # consumed via paired weight

            t = raw[k]
            new_name = remap(k)

            if k in weight_keys:
                scale_k = k[: -len(".weight")] + ".scale"
                if scale_k in all_scales:
                    s = all_scales[scale_k]
                    if t.dtype == torch.int8:
                        out = dequant_mxfp4(t, s)
                    elif t.dtype == torch.float8_e4m3fn:
                        out = dequant_block_fp8(t, s)
                    else:
                        raise ValueError(f"Unexpected weight dtype {t.dtype} for key {k}")
                    new_state[new_name] = out.cpu()
                    continue
                if t.dtype in (torch.int8, torch.float8_e4m3fn):
                    raise ValueError(f"Quantized weight {k} has no scale (looking for {scale_k})")

            new_state[new_name] = t.cpu()

        out_path = os.path.join(bf16_path, shard_name)
        save_file(new_state, out_path)
        for new_name in new_state:
            new_weight_map[new_name] = shard_name

        del raw, new_state
        torch.cuda.empty_cache()

    # Write updated index.json
    new_index = {"metadata": {"total_size": sum(os.path.getsize(os.path.join(bf16_path, f)) for f in {v for v in new_weight_map.values()})}, "weight_map": new_weight_map}
    with open(os.path.join(bf16_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # Copy aux files
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(flash_path, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(bf16_path, fname))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-flash-hf-path", required=True)
    parser.add_argument("--output-bf16-hf-path", required=True)
    args = parser.parse_args()
    main(args.input_flash_hf_path, args.output_bf16_hf_path)
