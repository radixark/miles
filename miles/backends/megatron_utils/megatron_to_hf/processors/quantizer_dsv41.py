"""DeepSeek-V4.1 weight sync in the layouts SGLang serves the release in, with FP32 block scales.

The release stores its dense projections as E4M3 with 32x32 UE8M0 scales and its routed experts as MXFP4. On Hopper,
SGLang serves the dense projections in that 32x32 layout (DeepGEMM SM90 reads FP32 scales), dequantizes ``wo_a`` to BF16
at load, and, under ``SGLANG_DSV4_FP4_DEQUANT``, casts the experts to E4M3 with 128x128 scales at load.

Sync requantizes the BF16 masters into those layouts with FP32 ``amax / 448`` scales. The masters are the exact dequant
of the release, so requantizing them with the release's power-of-two rule would map small updates back to the release
bytes; FP32 scales do not. ``wo_a`` travels in BF16, the precision serving runs it in.
"""

import os
import re

import torch

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
# The release's 32x32 FP8 projections, by checkpoint name.
_DENSE = re.compile(
    r"layers\.\d+\.(attn\.(wq_a|wq_b|wkv|wo_b|indexer\.wq_b)|ffn\.shared_experts\.w[123]|engram\.wkv)\.weight"
)
_EXPERT = re.compile(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight")
DENSE_BLOCK = (32, 32)
# SGLANG_DSV4_FP4_DEQUANT's layout: cast_e2m1fn_to_e4m3fn writes 128x128 blocks.
EXPERT_BLOCK = (128, 128)


def quantize_params_dsv41(converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    assert quantization_config["quant_method"] == "fp8" and list(quantization_config["weight_block_size"]) == list(
        DENSE_BLOCK
    ), f"DeepSeek-V4.1 sync expects the release's 32x32 FP8 config, got {quantization_config}"
    out = []
    for name, param in converted_named_params:
        if _DENSE.fullmatch(name):
            out += _fp8_pair(name, param, DENSE_BLOCK)
        elif _EXPERT.fullmatch(name):
            _check_expert_cast(quantization_config)
            out += _fp8_pair(name, param, EXPERT_BLOCK)
        else:
            out.append((name, param))
    return out


def block_cast_fp32(weight: torch.Tensor, block: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """E4M3 codes and one FP32 ``amax / 448`` scale per block; the dimensions must be block multiples."""
    rows, cols = weight.shape
    bm, bn = block
    assert rows % bm == 0 and cols % bn == 0, f"{tuple(weight.shape)} is not a multiple of {block}"
    blocks = weight.float().view(rows // bm, bm, cols // bn, bn)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-10) / _FP8_MAX
    codes = (blocks / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return codes.view(rows, cols), scale.view(rows // bm, cols // bn)


def _fp8_pair(name, param, block):
    weight, scale = block_cast_fp32(param, block)
    return [(name, weight), (name.removesuffix(".weight") + ".weight_scale_inv", scale)]


def _check_expert_cast(quantization_config):
    if quantization_config.get("expert_dtype") != "fp4":
        raise NotImplementedError("DeepSeek-V4.1 sync sends FP8 experts only for the release's MXFP4 experts")
    if os.environ.get("SGLANG_DSV4_FP4_DEQUANT", "").lower() not in ("1", "true"):
        raise RuntimeError(
            "DeepSeek-V4.1 sync sends the routed experts as 128x128 FP8, the layout serving casts them to under "
            "SGLANG_DSV4_FP4_DEQUANT=1; set it for the trainer and the engines"
        )
