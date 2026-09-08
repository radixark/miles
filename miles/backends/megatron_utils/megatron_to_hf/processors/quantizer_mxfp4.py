"""Packed MXFP4 quantization for DeepSeek-V4 routed experts."""

import re

from miles.utils.mxfp4 import mxfp4_quantize

_ROUTED_EXPERT_LINEARS = ("linear_fc1", "linear_fc2")


def is_routed_expert_param(megatron_name: str) -> bool:
    """Whether a Megatron parameter name addresses a routed expert linear."""
    match = re.search(r"(?:decoder|mtp)\.layers\.\d+\.(.+)", megatron_name)
    if not match:
        return False
    rest = match.group(1).replace("transformer_layer.", "").replace("mtp_model_layer.", "")
    expert_match = re.match(r"mlp\.experts\.(.+)\.weight\d+", rest)
    return bool(expert_match) and expert_match.group(1) in _ROUTED_EXPERT_LINEARS


def quantize_params_mxfp4(converted_named_params):
    """Quantize routed-expert weights to packed MXFP4 with UE8M0 block scales."""
    quantized_named_params = []
    for converted_name, param in converted_named_params:
        if converted_name.endswith("_scale"):
            continue
        if not converted_name.endswith(".weight"):
            raise ValueError(f"Expected weight parameter, got {converted_name}")
        qweight, scale = mxfp4_quantize(param)
        quantized_named_params.extend(
            [
                (converted_name, qweight),
                (converted_name.replace(".weight", ".weight_scale_inv"), scale),
            ]
        )
    return quantized_named_params
