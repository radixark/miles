"""Megatron(torch_dist) -> HuggingFace converter for MiniMax-M3.

Inverse of the mbridge bridge (miles_plugins/mbridge/minimax_m3.py) used for the
HF->torch_dist direction (which was validated bit-exact offline). Used by
tools/convert_torch_dist_to_hf.py to export a TRAINED M3 checkpoint back to HF.

Params arrive TP-merged and per-expert (``module.module.decoder.layers.{i}.{rest}``,
experts already split to ``mlp.experts.linear_fc1.weight{idx}`` by get_expert_param).
HF keys mirror the real M3 VL checkpoint layout: prefix ``language_model.model.*``,
lm_head ``language_model.lm_head.weight`` (so the export drops into the same
MiniMaxM3VL arch, with the vision tower carried over separately).

M3 vs qwen3moe differences handled here:
  * VLM-nested HF prefix ``language_model.model`` (not ``model``).
  * MoE experts -> ``block_sparse_moe.experts.{e}.{w1,w3,w2}`` (gate=w1, up=w3, down=w2).
  * Router -> ``block_sparse_moe.gate.weight`` (+ ``e_score_correction_bias``).
  * Shared experts -> ``block_sparse_moe.shared_experts.{gate,up,down}_proj``.
  * MSA lightning indexer (sparse layers) -> ``self_attn.index_{q,k}_{proj,norm}`` (direct).
  * Dense layers 0-2 -> plain ``mlp.{gate,up,down}_proj``.
  * No attention bias (--disable-bias-linear). gemma/zero-centered norms map directly
    (megatron stores gamma == HF (1+w) gamma; confirmed bit-exact on final_norm).
"""

import re

import torch

_P = "language_model.model"


def convert_minimax_m3_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [(f"{_P}.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("language_model.lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [(f"{_P}.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unknown parameter name: {name}")
    layer_idx, rest = match.groups()
    pfx = f"{_P}.layers.{layer_idx}"

    # --- MoE routed experts (grouped GEMM, one weight per expert) ---
    em = re.match(r"mlp\.experts\.(.+)\.weight(\d+)", rest)
    if em:
        sub, expert_idx = em.groups()
        ep = f"{pfx}.block_sparse_moe.experts.{expert_idx}"
        if sub == "linear_fc1":
            gate_weight, up_weight = param.chunk(2, dim=0)  # w1=gate, w3=up
            return [(f"{ep}.w1.weight", gate_weight), (f"{ep}.w3.weight", up_weight)]
        if sub == "linear_fc2":
            return [(f"{ep}.w2.weight", param)]
        raise ValueError(f"Unknown expert parameter name: {name}")

    # --- shared expert ---
    sm = re.match(r"mlp\.shared_experts\.(.+)", rest)
    if sm:
        sub = sm.groups()[0]
        sp = f"{pfx}.block_sparse_moe.shared_experts"
        if sub == "linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [(f"{sp}.gate_proj.weight", gate_weight), (f"{sp}.up_proj.weight", up_weight)]
        if sub == "linear_fc2.weight":
            return [(f"{sp}.down_proj.weight", param)]
        raise ValueError(f"Unknown shared expert parameter name: {name}")

    # --- attention (GQA, no bias) ---
    if rest == "self_attention.linear_proj.weight":
        return [(f"{pfx}.self_attn.o_proj.weight", param)]
    if rest == "self_attention.linear_qkv.weight":
        param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
        q_param, k_param, v_param = torch.split(param, [value_num_per_group, 1, 1], dim=1)
        return [
            (f"{pfx}.self_attn.q_proj.weight", q_param.reshape(-1, args.hidden_size)),
            (f"{pfx}.self_attn.k_proj.weight", k_param.reshape(-1, args.hidden_size)),
            (f"{pfx}.self_attn.v_proj.weight", v_param.reshape(-1, args.hidden_size)),
        ]
    if rest == "self_attention.linear_qkv.layer_norm_weight":
        return [(f"{pfx}.input_layernorm.weight", param)]
    if rest == "self_attention.q_layernorm.weight":
        return [(f"{pfx}.self_attn.q_norm.weight", param)]
    if rest == "self_attention.k_layernorm.weight":
        return [(f"{pfx}.self_attn.k_norm.weight", param)]
    # MSA lightning indexer (sparse layers 3-59) — direct name map, no reshape
    if rest == "self_attention.index_q_proj.weight":
        return [(f"{pfx}.self_attn.index_q_proj.weight", param)]
    if rest == "self_attention.index_k_proj.weight":
        return [(f"{pfx}.self_attn.index_k_proj.weight", param)]
    if rest == "self_attention.index_q_norm.weight":
        return [(f"{pfx}.self_attn.index_q_norm.weight", param)]
    if rest == "self_attention.index_k_norm.weight":
        return [(f"{pfx}.self_attn.index_k_norm.weight", param)]

    # --- dense MLP (layers 0-2) ---
    if rest == "mlp.linear_fc1.weight":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [(f"{pfx}.mlp.gate_proj.weight", gate_weight), (f"{pfx}.mlp.up_proj.weight", up_weight)]
    if rest == "mlp.linear_fc2.weight":
        return [(f"{pfx}.mlp.down_proj.weight", param)]
    if rest == "mlp.linear_fc1.layer_norm_weight":
        return [(f"{pfx}.post_attention_layernorm.weight", param)]

    # --- MoE router + pre-mlp norm (layers 3-59) ---
    if rest == "pre_mlp_layernorm.weight":
        return [(f"{pfx}.post_attention_layernorm.weight", param)]
    if rest == "mlp.router.weight":
        return [(f"{pfx}.block_sparse_moe.gate.weight", param)]
    if rest == "mlp.router.expert_bias":
        return [(f"{pfx}.block_sparse_moe.e_score_correction_bias", param)]

    raise ValueError(f"Unknown parameter name: {name}")
