import re

from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize


def quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config):
    assert quantization_config["quant_method"] == "mxfp8"

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.search(decoder_layers_pattern, megatron_name)

    if not match:
        # check mtp layers
        mtp_layer_pattern = r"mtp\.layers\.(\d+)\.(.+)"
        match = re.search(mtp_layer_pattern, megatron_name)
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
                quantize_named_params.extend(_quantize_param(converted_name, param))

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
                quantize_named_params.extend(_quantize_param(converted_name, param))

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
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(_quantize_param(converted_name, param))

        return quantize_named_params

    # for other parameters, we just return the original converted_named_params
    return converted_named_params


def _quantize_param(name, weight):
    if mxfp8_group_quantize is None:
        raise RuntimeError("MXFP8 quantization requires sglang fp8_utils.mxfp8_group_quantize.")
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % 32 != 0:
        raise ValueError(f"Last dim {k} must be divisible by 32 for MXFP8.")
    weight_flat = weight.view(-1, k).contiguous()
    qweight, scale = mxfp8_group_quantize(weight_flat)
    qweight = qweight.view_as(weight)
    scale = scale.view(*weight.shape[:-1], k // 32).contiguous()
    scale_name = name.replace(".weight", ".weight_scale_inv")
    return [(name, qweight), (scale_name, scale)]
