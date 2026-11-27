import re

import torch

from miles.utils.mxfp8 import mxfp8_quantize
from miles.utils.nvfp4 import FP4_E2M1_MAX, nvfp4_quantize_1d, nvfp4_quantize_1d_pair, nvfp4_weight_e4m3_max

# One expert's projections from convert_inkling2_to_hf.
_EXPERT_PARAM = re.compile(r"^(model\.llm\.layers\.\d+\.mlp\.experts\.)\d+\.(gate_proj|up_proj|down_proj)\.weight$")


def quantize_params_inkling2(converted_named_params, quantization_config):
    """Encode one expert the way Inkling2's checkpoint does (hf_quant_config.json), so the engine
    serves trained weights at the release precision. NVFP4 interleaves gate and up rows under one
    global amax (W = e2m1 * block_scale * amax / denominator); MXFP8 keeps a ue8m0 scale per 32."""
    match = _EXPERT_PARAM.match(converted_named_params[0][0])
    if match is None:
        return converted_named_params
    stack, proj = match.groups()
    expert_prefix = converted_named_params[0][0].removesuffix(f"{proj}.weight")
    leaf = "w2_weight" if proj == "down_proj" else "w13_weight"
    compute = quantization_config["expert_formats"][stack + leaf]
    if compute == "bfloat16":
        return converted_named_params
    params = [param for _, param in converted_named_params]
    if compute == "nvfp4":
        if leaf == "w13_weight":
            (gate, gate_scale, decode), (up, up_scale, _) = nvfp4_quantize_1d_pair(*params)
            weight, scale = _interleave_rows(gate, up), _interleave_rows(gate_scale, up_scale)
        else:
            weight, scale, decode = nvfp4_quantize_1d(params[0])
        denominator = float(nvfp4_weight_e4m3_max()) * FP4_E2M1_MAX
        return [
            (expert_prefix + leaf, weight),
            (expert_prefix + leaf + "_scale", scale),
            (expert_prefix + leaf + "_amax", (decode * denominator).reshape(())),
            (expert_prefix + leaf + "_outer_scale_denominator", torch.tensor(denominator, device=weight.device)),
        ]
    assert compute == "mxfp8", f"{stack + leaf}: unsupported Inkling2 expert precision {compute}"
    quantized = [mxfp8_quantize(param) for param in params]
    if leaf == "w13_weight":
        (gate, gate_scale), (up, up_scale) = quantized
        weight, scale = _interleave_rows(gate, up), _interleave_rows(gate_scale, up_scale)
    else:
        weight, scale = quantized[0]
    return [(expert_prefix + leaf, weight), (expert_prefix + leaf + "_scale", scale)]


def _interleave_rows(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """[f0, s0, f1, s1, ...]: the engine's w13 row layout."""
    return torch.stack((first, second), dim=1).flatten(0, 1)
