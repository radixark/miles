import re

import torch

from miles.utils.mxfp8 import mxfp8_quantize
from miles.utils.nvfp4 import nvfp4_global_encode_scale_te, nvfp4_quantize_1d, nvfp4_quantize_1d_pair

# the M3.1 checkpoint: routed experts in compressed-tensors NVFP4, every other projection MXFP8
_EXPERT = re.compile(r"\.block_sparse_moe\.experts\.\d+\.(w[123])\.weight$")
_MXFP8 = re.compile(
    r"\.(self_attn\.(q|k|v|o|index_q|index_k)_proj"
    r"|mlp\.(gate|up|down)_proj"
    r"|block_sparse_moe\.shared_experts\.(gate|up|down)_proj)\.weight$"
)


def _nvfp4_entries(name: str, qweight, block_scale, encode_scale):
    base = name.removesuffix(".weight")
    return [
        (f"{base}.weight_packed", qweight),
        (f"{base}.weight_scale", block_scale),
        (f"{base}.weight_global_scale", encode_scale.reshape(()).to(torch.float32)),
    ]


def quantize_params_minimax_m3_nvfp4(converted_named_params):
    experts = {}
    quantized = []
    for name, param in converted_named_params:
        if (m := _EXPERT.search(name)) is not None:
            experts[m[1]] = (name, param)
        elif _MXFP8.search(name) is not None:
            qweight, scale = mxfp8_quantize(param)
            quantized += [(name, qweight), (name.replace(".weight", ".weight_scale_inv"), scale)]
        else:
            quantized.append((name, param))

    if "w2" in experts:
        name, param = experts.pop("w2")
        qweight, block_scale, _ = nvfp4_quantize_1d(param)
        quantized += _nvfp4_entries(name, qweight, block_scale, nvfp4_global_encode_scale_te(param.abs().amax()))
    if experts:
        assert set(experts) == {"w1", "w3"}, f"gate and up must convert together, got {sorted(experts)}"
        (gate_name, gate), (up_name, up) = experts["w1"], experts["w3"]
        # gate and up share one encode scale, as in the checkpoint
        encode = nvfp4_global_encode_scale_te(torch.maximum(gate.abs().amax(), up.abs().amax()))
        (gq, gs, _), (uq, us, _) = nvfp4_quantize_1d_pair(gate, up)
        quantized += _nvfp4_entries(gate_name, gq, gs, encode) + _nvfp4_entries(up_name, uq, us, encode)
    return quantized
