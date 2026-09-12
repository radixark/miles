"""Megatron -> HF weight conversion for GLM-5.3-Flash (glm5_next), the inverse
of ``miles_plugins/mbridge/glm5_next.py`` for the weight-update direction.

Everything DeepseekV3-shaped delegates to ``convert_deepseekv3_to_hf``; the DSA
indexer names are handled here instead so the converter's rope-interleave
half-swap can never run (GLM-5.3 has ``qk_rope_head_dim == 0``). The packed
``kda.conv1d.weight`` splits into the checkpoint's ``{q,k,v}_conv1d.weight``;
the three ``alpha_*`` parameters (always on the same rank) are buffered per
(layer, site) and emitted as one ``hc_*_scale`` tensor once all have arrived.
"""

import re

import torch

from .deepseekv3 import convert_deepseekv3_to_hf

_KDA_SUFFIX_MAPPING = {
    f"self_attention.kda.{weight_name}": f"self_attn.{weight_name}"
    for weight_name in [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "b_proj.weight",
        "f_a_proj.weight",
        "f_b_proj.weight",
        "g_a_proj.weight",
        "g_b_proj.weight",
        "A_log",
        "dt_bias",
        "o_norm.weight",
        "o_proj.weight",
    ]
}

_INDEXER_SUFFIX_MAPPING = {
    "self_attention.wq_b.weight": "self_attn.indexer.wq_b.weight",
    "self_attention.wk.weight": "self_attn.indexer.wk.weight",
    "self_attention.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
    "self_attention.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "self_attention.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "self_attention.index_kpool_compress_gate": "self_attn.indexer.index_kpool_compress_gate",
    "self_attention.index_kpool_compress_ape": "self_attn.indexer.index_kpool_compress_ape",
}

_HC_SUFFIX_MAPPING = {
    "self_attention_hyper_connection.mapping_proj.weight": "hc_attn_fn",
    "self_attention_hyper_connection.bias": "hc_attn_base",
    "mlp_hyper_connection.mapping_proj.weight": "hc_ffn_fn",
    "mlp_hyper_connection.bias": "hc_ffn_base",
}

_HC_ALPHA_ORDER = ("alpha_pre", "alpha_post", "alpha_res")
_HC_SITE_TO_SCALE = {
    "self_attention_hyper_connection": "hc_attn_scale",
    "mlp_hyper_connection": "hc_ffn_scale",
}

_LAYER_PATTERN = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.(.+)")
_HC_ALPHA_PATTERN = re.compile(
    r"(self_attention_hyper_connection|mlp_hyper_connection)\.(alpha_pre|alpha_post|alpha_res)$"
)

_hc_scale_buffers: dict[tuple[str, str], dict[str, torch.Tensor]] = {}


def _convert_hc_scale(layer_idx: str, site: str, alpha_name: str, param: torch.Tensor):
    buffer = _hc_scale_buffers.setdefault((layer_idx, site), {})
    buffer[alpha_name] = param
    if len(buffer) < len(_HC_ALPHA_ORDER):
        return []
    _hc_scale_buffers.pop((layer_idx, site))
    scale = torch.cat([buffer[name].reshape(1) for name in _HC_ALPHA_ORDER])
    return [(f"model.layers.{layer_idx}.{_HC_SITE_TO_SCALE[site]}", scale)]


def convert_glm5_next_to_hf(args, name, param):
    match = _LAYER_PATTERN.match(name)
    if match:
        layer_idx, rest = match.groups()

        hf_suffix = _KDA_SUFFIX_MAPPING.get(rest) or _INDEXER_SUFFIX_MAPPING.get(rest) or _HC_SUFFIX_MAPPING.get(rest)
        if hf_suffix is not None:
            return [(f"model.layers.{layer_idx}.{hf_suffix}", param)]

        if rest == "self_attention.kda.conv1d.weight":
            q_conv, k_conv, v_conv = param.chunk(3, dim=0)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_conv1d.weight", q_conv),
                (f"model.layers.{layer_idx}.self_attn.k_conv1d.weight", k_conv),
                (f"model.layers.{layer_idx}.self_attn.v_conv1d.weight", v_conv),
            ]

        alpha_match = _HC_ALPHA_PATTERN.match(rest)
        if alpha_match:
            site, alpha_name = alpha_match.groups()
            return _convert_hc_scale(layer_idx, site, alpha_name, param)

    return convert_deepseekv3_to_hf(args, name, param)
