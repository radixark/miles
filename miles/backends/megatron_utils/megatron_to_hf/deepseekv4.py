import re

import torch


def convert_deepseekv4_to_hf(args, name, param):
    # Embedding and output
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    # Block-level Hyper-Connection weights
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

        # Layer-level Hyper-Connection weights
        if rest == "hc_attn_fn":
            return [(f"model.layers.{layer_idx}.hc_attn_fn", param)]
        elif rest == "hc_attn_base":
            return [(f"model.layers.{layer_idx}.hc_attn_base", param)]
        elif rest == "hc_attn_scale":
            return [(f"model.layers.{layer_idx}.hc_attn_scale", param)]
        elif rest == "hc_ffn_fn":
            return [(f"model.layers.{layer_idx}.hc_ffn_fn", param)]
        elif rest == "hc_ffn_base":
            return [(f"model.layers.{layer_idx}.hc_ffn_base", param)]
        elif rest == "hc_ffn_scale":
            return [(f"model.layers.{layer_idx}.hc_ffn_scale", param)]

        # Experts
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

        # Shared expert
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

        # V4 Attention weights - use MQALayer param names directly
        if rest == "self_attention.wq_a.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wq_a.weight", param)]
        elif rest == "self_attention.q_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.wq_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wq_b.weight", param)]
        elif rest == "self_attention.wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wkv.weight", param)]
        elif rest == "self_attention.kv_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_norm.weight", param)]
        elif rest == "self_attention.wo_a.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wo_a.weight", param)]
        elif rest == "self_attention.wo_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wo_b.weight", param)]
        elif rest == "self_attention.attn_sink":
            return [(f"model.layers.{layer_idx}.self_attn.attn_sink", param)]

        # Compressor weights
        elif rest == "self_attention.compressor.ape":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.ape", param)]
        elif rest == "self_attention.compressor.wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wkv.weight", param)]
        elif rest == "self_attention.compressor.wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wgate.weight", param)]
        elif rest == "self_attention.compressor.norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.norm.weight", param)]

        # DSA Indexer weights
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

        # Layernorms
        elif rest == "input_layernorm.weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # MoE router
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]
        elif rest == "mlp.router.tid2eid":
            return [(f"model.layers.{layer_idx}.mlp.topk.tid2eid", param)]

        # Dense MLP (if any layer is not MoE)
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
