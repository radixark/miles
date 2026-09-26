"""Weight-sync quantizer for FP8 block checkpoints that store the routed experts as MXFP4.

`quant_method: fp8` with `store_dtype: mxfp4` (MiMo-V2): routed experts are e2m1 nibbles with E8M0
scales (`weight` + `weight_scale`), the modules in `ignored_layers` stay BF16, and everything else
follows quantize_params_fp8. A fused `qkv_proj` is block-quantized per kv-head shard, as stored in
the checkpoint and sliced by SGLang; on Blackwell its scales take the same UE8M0 layout as every
other dense FP8 weight (SGLang requantizes them to it at load).
"""

import json
import os
import re
from functools import lru_cache

import torch

from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import _quantize_param, quantize_params_fp8
from miles.utils.mxfp4 import quantize_mxfp4

_ROUTED_EXPERT_RE = re.compile(
    r"\.mlp\.experts\.(linear_fc[12]\.weight\d+|local_experts\.\d+\.linear_fc[12]\.weight)$"
)


def quantize_params_fp8_mxfp4_experts(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "fp8" and quantization_config["store_dtype"] == "mxfp4"
    ignored = set(quantization_config.get("ignored_layers") or ())
    if all(name.removesuffix(".weight") in ignored for name, _ in converted_named_params):
        return converted_named_params

    if _ROUTED_EXPERT_RE.search(megatron_name):
        group_size = quantization_config["mxfp4_block_size"]
        quantized = []
        for name, param in converted_named_params:
            assert name.endswith(".weight"), f"unexpected routed-expert tensor {name}"
            packed, scale = quantize_mxfp4(param, group_size)
            quantized += [(name, packed), (f"{name.removesuffix('.weight')}.weight_scale", scale)]
        return quantized

    if any(name.endswith(".self_attn.qkv_proj.weight") for name, _ in converted_named_params):
        assert len(converted_named_params) == 1, f"{megatron_name} exports more than the fused qkv_proj"
        ((name, param),) = converted_named_params
        return _quantize_fused_qkv(
            args, name, param, _fused_qkv_shards(args.hf_checkpoint), quantization_config["weight_block_size"]
        )

    return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)


@lru_cache(maxsize=4)
def _fused_qkv_shards(hf_checkpoint: str) -> int:
    # SGLang slices a fused MiMo-V2 qkv_proj into num_key_value_heads shards
    # (get_mimo_v2_fused_qkv_expected_tp_size).
    with open(os.path.join(hf_checkpoint, "config.json")) as f:
        return json.load(f)["num_key_value_heads"]


def _quantize_fused_qkv(args, name, weight, shards, weight_block_size):
    """Each shard gets its own FP8 blocks, the last row block of a shard left partial."""
    assert weight.shape[0] % shards == 0, f"{name}: {weight.shape[0]} rows do not split into {shards} shards"
    shard_rows = weight.shape[0] // shards
    pieces = [_quantize_param(args, name, shard, weight_block_size) for shard in weight.chunk(shards)]
    (_, _), (scale_name, first_scale) = pieces[0]
    # Either scale layout stacks along rows, which is how SGLang slices the shards back: fp32 holds one
    # row per 128-row block of a shard, the DeepGEMM UE8M0 layout (int32, Blackwell) one row per weight row.
    scale_rows = shard_rows if first_scale.dtype == torch.int32 else -(-shard_rows // weight_block_size[0])
    for (_, _), (_, shard_scale) in pieces:
        assert shard_scale.shape[0] == scale_rows, f"{name}: shard scale {tuple(shard_scale.shape)}, {scale_rows} rows"
    qweight = torch.cat([piece[0][1] for piece in pieces])
    scale = torch.cat([piece[1][1] for piece in pieces])
    return [(name, qweight), (scale_name, scale)]
