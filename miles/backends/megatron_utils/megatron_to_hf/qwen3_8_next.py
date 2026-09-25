"""Megatron -> HF weight conversion for Qwen3.8-Flash-Next (Qwen4Exp).

Inverse of the HF -> mcore names in ``Qwen38NextBridge``; everything Qwen3.5-shaped
delegates to ``convert_qwen3_5_to_hf``.
"""

import re

from .qwen3_5 import convert_qwen3_5_to_hf

# mcore suffix (under decoder.layers.N.) -> HF suffix (under the HF layer prefix)
_HC_SUFFIX_MAPPING = {
    "self_attention_hyper_connection.hc_norm_weight": "attn_hyper_connection.hc_norm.weight",
    "self_attention_hyper_connection.input_mix_weight_down": "attn_hyper_connection.input_mix_weight_down.weight",
    "self_attention_hyper_connection.input_mix_weight_up": "attn_hyper_connection.input_mix_weight_up.weight",
    "self_attention_hyper_connection.block_inject_weight": "attn_hyper_connection.block_inject_weight.weight",
    "mlp_hyper_connection.hc_norm_weight": "mlp_hyper_connection.hc_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down": "mlp_hyper_connection.input_mix_weight_down.weight",
    "mlp_hyper_connection.input_mix_weight_up": "mlp_hyper_connection.input_mix_weight_up.weight",
    "mlp_hyper_connection.block_inject_weight": "mlp_hyper_connection.block_inject_weight.weight",
    # the frozen n-gram table is not a parameter, so it never goes through weight sync
    "self_attention_hyper_connection.ple.key_proj.weight": "ple.key_proj.weight",
    "self_attention_hyper_connection.ple.value_proj.weight": "ple.value_proj.weight",
    "self_attention_hyper_connection.ple.conv1d_weight": "ple.conv1d.weight",
    "self_attention_hyper_connection.ple.norm_conv": "ple.norm_conv.weight",
    "self_attention_hyper_connection.ple.norm_key": "ple.norm_key.weight",
    "self_attention_hyper_connection.ple.norm_query": "ple.norm_query.weight",
    "self_attention.indexer.index_qk_proj.weight": "self_attn.indexer.index_qk_proj.weight",
    "self_attention.indexer.q_layernorm": "self_attn.indexer.q_layernorm.weight",
    "self_attention.indexer.k_layernorm": "self_attn.indexer.k_layernorm.weight",
}

# the final mixer's hc_norm stands in for decoder.final_layernorm
_HEAD_CONTRACTION_MAPPING = {
    "module.module.decoder.hc_head_contraction.hc_norm_weight": "model.language_model.hyper_connection_mixer.hc_norm.weight",
    "module.module.decoder.hc_head_contraction.input_mix_weight_down": "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
    "module.module.decoder.hc_head_contraction.input_mix_weight_up": "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
}

_LAYER_PATTERN = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.(.+)")


def convert_qwen3_8_next_to_hf(args, name, param):
    if name in _HEAD_CONTRACTION_MAPPING:
        return [(_HEAD_CONTRACTION_MAPPING[name], param)]

    match = _LAYER_PATTERN.match(name)
    if match:
        layer_idx, rest = match.groups()
        hf_suffix = _HC_SUFFIX_MAPPING.get(rest)
        if hf_suffix is not None:
            return [(f"model.language_model.layers.{layer_idx}.{hf_suffix}", param)]

    return convert_qwen3_5_to_hf(args, name, param)
