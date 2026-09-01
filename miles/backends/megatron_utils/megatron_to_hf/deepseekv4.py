import re

import torch

from ..update_weight.common import AtomicUpdateGroup


def get_deepseek_v4_atomic_update_groups():
    return [
        AtomicUpdateGroup(key, suffixes)
        for key, suffixes in [
            ("wqkv_a", (".self_attention.linear_q_down_proj.weight", ".self_attention.linear_kv_proj.weight")),
            (
                "self_attention_hc_alphas",
                (
                    ".self_attention_hyper_connection.alpha_pre",
                    ".self_attention_hyper_connection.alpha_post",
                    ".self_attention_hyper_connection.alpha_res",
                ),
            ),
            (
                "mlp_hc_alphas",
                (
                    ".mlp_hyper_connection.alpha_pre",
                    ".mlp_hyper_connection.alpha_post",
                    ".mlp_hyper_connection.alpha_res",
                ),
            ),
            (
                "compressor_wkv_gate",
                (
                    ".self_attention.core_attention.compressor.linear_wkv.weight",
                    ".self_attention.core_attention.compressor.linear_wgate.weight",
                ),
            ),
            (
                "indexer_compressor_wkv_gate",
                (
                    ".self_attention.core_attention.indexer.compressor.linear_wkv.weight",
                    ".self_attention.core_attention.indexer.compressor.linear_wgate.weight",
                ),
            ),
        ]
    ]


_PENDING_ALPHAS: dict[str, dict[str, torch.Tensor]] = {}


def _packed_alphas(name: str, param):
    """Pack a hyper-connection site's three alphas the way the checkpoint stores them.

    Megatron keeps alpha_pre / alpha_post / alpha_res as separate scalars, one per bias
    segment; the checkpoint carries them as one [pre, post, res] tensor. The three may
    arrive in any order across weight-sync buckets, so stash pieces and emit the packed
    tensor once the site is complete.
    """
    prefix, seg = name.rsplit(".alpha_", 1)
    slots = _PENDING_ALPHAS.setdefault(prefix, {})
    slots[seg] = param.reshape(1)
    if len(slots) < 3:
        return None
    del _PENDING_ALPHAS[prefix]
    return torch.cat([slots[seg] for seg in ("pre", "post", "res")])


def convert_deepseekv4_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    if name == "module.module.decoder.hc_head_fn":
        return [("model.hc_head_fn", param)]
    if name == "module.module.decoder.hc_head_base":
        return [("model.hc_head_base", param)]
    if name == "module.module.decoder.hc_head_scale":
        return [("model.hc_head_scale", param)]

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        if rest == "self_attention_hyper_connection.mapping_proj.weight":
            return [(f"model.layers.{layer_idx}.hc_attn_fn", param)]
        elif rest == "self_attention_hyper_connection.bias":
            return [(f"model.layers.{layer_idx}.hc_attn_base", param)]
        elif rest.startswith("self_attention_hyper_connection.alpha_"):
            packed = _packed_alphas(name, param)
            return [] if packed is None else [(f"model.layers.{layer_idx}.hc_attn_scale", packed)]
        elif rest == "mlp_hyper_connection.mapping_proj.weight":
            return [(f"model.layers.{layer_idx}.hc_ffn_fn", param)]
        elif rest == "mlp_hyper_connection.bias":
            return [(f"model.layers.{layer_idx}.hc_ffn_base", param)]
        elif rest.startswith("mlp_hyper_connection.alpha_"):
            packed = _packed_alphas(name, param)
            return [] if packed is None else [(f"model.layers.{layer_idx}.hc_ffn_scale", packed)]

        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wq_a.weight", param)]
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wq_b.weight", param)]
        elif rest == "self_attention.linear_kv_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wkv.weight", param)]
        elif rest == "self_attention.kv_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_norm.weight", param)]
        elif rest == "self_attention.linear_o_group_proj":
            return [(f"model.layers.{layer_idx}.self_attn.wo_a.weight", param)]
        elif rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wo_b.weight", param)]
        elif rest == "self_attention.core_attention.attn_sink":
            return [(f"model.layers.{layer_idx}.self_attn.attn_sink", param)]

        elif rest == "self_attention.core_attention.compressor.ape":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.ape", param)]
        elif rest == "self_attention.core_attention.compressor.linear_wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wkv.weight", param)]
        elif rest == "self_attention.core_attention.compressor.linear_wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wgate.weight", param)]
        elif rest == "self_attention.core_attention.compressor.norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.norm.weight", param)]

        elif rest == "self_attention.core_attention.indexer.linear_wq_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", param)]
        elif rest == "self_attention.core_attention.indexer.linear_wk.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wk.weight", param)]
        elif rest == "self_attention.core_attention.indexer.k_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight", param)]
        elif rest == "self_attention.core_attention.indexer.k_norm.bias":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias", param)]
        elif rest == "self_attention.core_attention.indexer.linear_weights_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight", param)]

        elif rest == "self_attention.core_attention.indexer.compressor.ape":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.ape", param)]
        elif rest == "self_attention.core_attention.indexer.compressor.linear_wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.wkv.weight", param)]
        elif rest == "self_attention.core_attention.indexer.compressor.linear_wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.wgate.weight", param)]
        elif rest == "self_attention.core_attention.indexer.compressor.norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.norm.weight", param)]

        elif rest == "input_layernorm.weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]
        elif rest == "mlp.router.tid2eid":
            return [(f"model.layers.{layer_idx}.mlp.topk.tid2eid", param)]

        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
