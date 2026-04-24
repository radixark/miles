import re

import torch


def _apply_ape_hotfix_mirror(param):
    # Mirror SGLang Compressor.apply_ape_hotfix for the 0415 checkpoint so that the
    # Megatron -> SGLang weight update lands in kernel layout. Logic: chunk ape on
    # the last dim (== ``2 * head_dim``), concat the two halves on dim 0, then view
    # back to ``(ratio=4, head_dim)``. Matches SGLang's default for ``ratio == 4``
    # under ``SGLANG_DSV4_MODE=2604`` (default) + ``SGLANG_OPT_FIX_APE_2604=True``
    # (default).
    assert param.shape[0] == 4
    ape = torch.chunk(param, 2, dim=-1)
    return torch.cat([ape[0], ape[1]], dim=0).view(4, -1).contiguous()


def convert_deepseekv4_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    if name == "module.module.decoder.hc_head_params.hc_head_fn":
        return [("model.hc_head_fn", param)]
    if name == "module.module.decoder.hc_head_params.hc_head_base":
        return [("model.hc_head_base", param)]
    if name == "module.module.decoder.hc_head_params.hc_head_scale":
        return [("model.hc_head_scale", param)]

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

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

        elif rest == "self_attention.compressor.ape":
            if param.shape[0] == 4:
                param = _apply_ape_hotfix_mirror(param)
            return [(f"model.layers.{layer_idx}.self_attn.compressor.ape", param)]
        elif rest == "self_attention.compressor.wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wkv.weight", param)]
        elif rest == "self_attention.compressor.wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wgate.weight", param)]
        elif rest == "self_attention.compressor.norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.norm.weight", param)]

        elif rest == "self_attention.indexer.linear_wq_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", param)]
        elif rest == "self_attention.indexer.linear_wk.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wk.weight", param)]
        elif rest == "self_attention.indexer.k_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight", param)]
        elif rest == "self_attention.indexer.k_norm.bias":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias", param)]
        elif rest == "self_attention.indexer.linear_weights_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight", param)]

        elif rest == "self_attention.indexer.compressor.ape":
            if param.shape[0] == 4:
                param = _apply_ape_hotfix_mirror(param)
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.ape", param)]
        elif rest == "self_attention.indexer.compressor.wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.wkv.weight", param)]
        elif rest == "self_attention.indexer.compressor.wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.wgate.weight", param)]
        elif rest == "self_attention.indexer.compressor.norm.weight":
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
