import re

import torch

from miles.utils.fp8_kernel import blockwise_cast_to_fp8_triton

from ...sglang import per_block_cast_to_fp8


def quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "fp8"
    fmt = quantization_config.get("fmt", "e4m3")
    assert fmt == "e4m3", f"Unsupported FP8 format: {fmt}"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size")

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
        match = re.match(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in [
            "linear_fc1",
            "linear_fc2",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(_quantize_weight(converted_name, param, weight_block_size))

            return quantize_named_params

    # shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in [
            "linear_fc1.weight",
            "linear_fc2.weight",
        ]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(_quantize_weight(converted_name, param, weight_block_size))

            return quantize_named_params

    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
        # DSA indexer
        "self_attention.wq_b.weight",
        "self_attention.wk.weight",
        # linear attention
        "self_attention.linear_attn.in_proj_qkv.weight",
        "self_attention.linear_attn.in_proj_z.weight",
        "self_attention.linear_attn.out_proj.weight",
        # DeepSeek V4 attention
        "self_attention.wq_a.weight",
        "self_attention.wkv.weight",
        "self_attention.wo_b.weight",
        "self_attention.indexer.linear_wq_b.weight",
        "self_attention.indexer.linear_wk.weight",
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(_quantize_weight(converted_name, param, weight_block_size))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_weight(name, weight, weight_block_size):
    # The engine derives backend-specific scale layouts while loading.
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if weight_block_size is not None:
        if per_block_cast_to_fp8 is not None and list(weight_block_size) == [128, 128]:
            qweight, scale = per_block_cast_to_fp8(weight)
        else:
            qweight, scale = blockwise_cast_to_fp8_triton(weight, weight_block_size)
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / fp8_max
        qweight = (weight / scale).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")
    return [(name, qweight), (scale_name, scale)]
