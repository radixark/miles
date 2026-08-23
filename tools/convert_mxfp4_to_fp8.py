"""
Convert an official DeepSeek-V4 checkpoint with MXFP4 routed experts to uniform
blockwise-FP8 (the sgl-project/DeepSeek-V4-*-FP8 layout).

Packed-e2m1fn expert weights (int8, with per-(1,32)-block ue8m0 scales) are cast
losslessly to e4m3fn with (128,128)-block e8m0 scales; all other tensors are
copied unchanged. `expert_dtype` is dropped from config.json.

python tools/convert_mxfp4_to_fp8.py --model-dir <mxfp4-hf> --save-dir <fp8-hf>
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# copied from deepseek-ai/DeepSeek-V4-Flash-0731 inference/convert.py
def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dtype == torch.int8 and x.ndim == 2
    out_dim, in_dim = x.size(0), x.size(1) * 2
    fp8_block_size, fp4_block_size = 128, 32
    assert out_dim % fp8_block_size == 0 and in_dim % fp8_block_size == 0
    assert scale.shape == (out_dim, in_dim // fp4_block_size), f"{scale.shape=} {x.shape=}"

    table = FP4_TABLE.to(x.device)
    x = x.view(torch.uint8)
    x = torch.stack([table[(x & 0x0F).long()], table[(x >> 4).long()]], dim=-1).reshape(out_dim, in_dim)

    max_offset = 2**6
    b_out, b_in = out_dim // fp8_block_size, in_dim // fp8_block_size
    x = x.view(b_out, fp8_block_size, b_in, fp8_block_size).transpose(1, 2)
    scale = scale.float().view(b_out, fp8_block_size, b_in, -1).transpose(1, 2).flatten(2)
    tile_scale = scale.amax(dim=-1, keepdim=True) / max_offset
    offset = (scale / tile_scale).unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), tile_scale.squeeze(-1).to(torch.float8_e8m0fnu)


def main(model_dir: str, save_dir: str, device: str):
    os.makedirs(save_dir, exist_ok=True)

    config = json.loads((Path(model_dir) / "config.json").read_text())
    config.pop("expert_dtype", None)
    if isinstance(config.get("quantization_config"), dict):
        config["quantization_config"].pop("expert_dtype", None)
    (Path(save_dir) / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for pattern in ("tokenizer*", "chat_template*", "generation_config.json"):
        os.system(f"cp -rf {model_dir}/{pattern} {save_dir}/ 2>/dev/null")

    index = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    weight_map = dict(index["weight_map"])
    shard_names = sorted(set(weight_map.values()))

    fp4_weights = set()
    for shard_name in shard_names:
        with safe_open(f"{model_dir}/{shard_name}", framework="pt") as f:
            for name in f.keys():
                if f.get_slice(name).get_dtype() == "I8":
                    assert name.endswith(".weight"), f"unexpected int8 tensor {name}"
                    fp4_weights.add(name)
    fp4_scales = {name.removesuffix(".weight") + ".scale" for name in fp4_weights}

    for shard_name in tqdm(shard_names):
        state_dict = load_file(f"{model_dir}/{shard_name}", device=device)
        new_state_dict = {}
        for name, tensor in state_dict.items():
            if name in fp4_weights:
                scale_name = name.removesuffix(".weight") + ".scale"
                scale = state_dict.get(scale_name)
                if scale is None:
                    scale = load_file(f"{model_dir}/{weight_map[scale_name]}", device=device)[scale_name]
                weight, tile_scale = cast_e2m1fn_to_e4m3fn(tensor, scale)
                new_state_dict[name] = weight
                new_state_dict[scale_name] = tile_scale
                weight_map[scale_name] = shard_name
            elif name in fp4_scales:
                continue
            else:
                new_state_dict[name] = tensor
        save_file(new_state_dict, f"{save_dir}/{shard_name}")

    (Path(save_dir) / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n"
    )
    assert fp4_weights, "no packed-fp4 expert weights found; is this already an FP8 checkpoint?"
    print(f"cast {len(fp4_weights)} expert weights to fp8, saved to {save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args.model_dir, args.save_dir, args.device)
