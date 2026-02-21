from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .llama import convert_llama_to_hf
from .mimo import convert_mimo_to_hf
from .processors import quantize_params, remove_padding
from .qwen2 import convert_qwen2_to_hf
from .qwen3_next import convert_qwen3_next_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


# TODO unify w/ `convert_to_hf`
def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    param = remove_padding(megatron_param_name, param, args.vocab_size)
    # TODO support quant
    return param


# TODO optimize code details
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    param = remove_padding(name, param, args.vocab_size)

    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)

    return quantize_params(args, name, converted_named_tensors, quantization_config)


# TODO optimize
_cached_tensors = {}


# TODO optimize code details
def _convert_to_hf_core(args, model_name, name, param):
    if "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "deepseekv3" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)

    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    elif "mimo" in model_name:
        converted_named_tensors = convert_mimo_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # to compatible with sglang implementation
    if args.q_lora_rank is not None:
        old_converted_named_tensors = converted_named_tensors
        converted_named_tensors = []
        for converted_name, converted_param in old_converted_named_tensors:
            if "q_a_proj" in converted_name:
                pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            elif "kv_a_proj_with_mqa" in converted_name:
                pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            else:
                converted_named_tensors.append((converted_name, converted_param))
    return converted_named_tensors


def convert_lora_to_hf(args, model_name, name, param):
    """Convert Megatron LoRA parameter to HuggingFace PEFT format.

    .. deprecated::
        This function uses hardcoded string replacements that do NOT correctly
        handle fused layers (e.g. linear_qkv -> q/k/v_proj, linear_fc1 ->
        gate/up_proj). Use ``AutoBridge.export_adapter_weights`` instead, which
        properly splits fused adapter weights and handles TP gathering.

    Megatron format: module.module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight
    HF PEFT format:  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    """
    import warnings

    warnings.warn(
        "convert_lora_to_hf uses incorrect hardcoded name mapping for fused layers. "
        "Use AutoBridge.export_adapter_weights instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Determine lora_A vs lora_B
    if ".linear_in." in name or ".lora_A." in name:
        lora_suffix = "lora_A.weight"
    elif ".linear_out." in name or ".lora_B." in name:
        lora_suffix = "lora_B.weight"
    else:
        return [(name, param)]

    # Convert Megatron naming to HF PEFT naming
    hf_name = name
    hf_name = hf_name.replace("module.module.", "base_model.model.")
    hf_name = hf_name.replace(".decoder.layers.", ".model.layers.")
    hf_name = hf_name.replace(".self_attention.linear_qkv", ".self_attn.q_proj")
    hf_name = hf_name.replace(".self_attention.linear_proj", ".self_attn.o_proj")
    hf_name = hf_name.replace(".mlp.linear_fc1", ".mlp.gate_proj")
    hf_name = hf_name.replace(".mlp.linear_fc2", ".mlp.down_proj")

    hf_name = hf_name.replace(".adapter.linear_in.weight", f".{lora_suffix}")
    hf_name = hf_name.replace(".adapter.linear_out.weight", f".{lora_suffix}")
    hf_name = hf_name.replace(".lora_A.weight", f".{lora_suffix}")
    hf_name = hf_name.replace(".lora_B.weight", f".{lora_suffix}")

    return [(hf_name, param)]
