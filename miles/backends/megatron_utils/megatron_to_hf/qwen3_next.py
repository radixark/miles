import re
from argparse import Namespace

import torch

from miles_plugins.mbridge.qwen3_next import convert_gated_attn_qgkv_mcore_to_hf, convert_gdn_in_proj_mcore_to_hf


def needs_gdn_weight_fix(name: str) -> bool:
    """Check if a parameter name requires GDN weight gather fix."""
    if "self_attention.in_proj.weight" in name and "layer_norm" not in name:
        return True
    if "self_attention.conv1d.weight" in name:
        return True
    if "self_attention.conv1d.bias" in name:
        return True
    return False


def fix_gdn_weight_gather(args: Namespace, name: str, param: torch.Tensor) -> torch.Tensor:
    """Fix GDN (GatedDeltaNet) per-component TP gathering.

    MCore's sharded_state_dict loads GDN in_proj and conv1d weights with per-component
    TP sharding: each rank holds [Q_local, K_local, V_local, ...].  But partition_stride=1
    causes all_gather_param to do a simple cat, producing
        [Q_r0, K_r0, V_r0, ..., Q_r1, K_r1, V_r1, ...]
    instead of the correct layout
        [Q_all, K_all, V_all, ...].
    This function rearranges the gathered tensor to the correct component-contiguous layout.
    """
    from megatron.core import mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return param

    qk_local = args.linear_num_key_heads * args.linear_key_head_dim // tp_size
    v_local = args.linear_num_value_heads * args.linear_value_head_dim // tp_size
    nv_local = args.linear_num_value_heads // tp_size

    if "self_attention.in_proj.weight" in name and "layer_norm" not in name:
        sections = [qk_local, qk_local, v_local, v_local, nv_local, nv_local]
    elif "self_attention.conv1d.weight" in name:
        sections = [qk_local, qk_local, v_local]
    elif "self_attention.conv1d.bias" in name:
        sections = [qk_local, qk_local, v_local]
    else:
        return param

    chunks = param.chunk(tp_size, dim=0)
    per_rank_comps = [c.split(sections, dim=0) for c in chunks]
    return torch.cat(
        [torch.cat([per_rank_comps[r][c] for r in range(tp_size)], dim=0) for c in range(len(sections))],
        dim=0,
    )


def handle_gdn_weight_gather(args: Namespace, name: str, param: torch.Tensor) -> torch.Tensor:
    """Apply GDN weight gather fix if needed (called after TP all-gather)."""
    if needs_gdn_weight_fix(name):
        param = fix_gdn_weight_gather(args, name, param)
    return param


def _convert_qgkv_weight_to_hf(args, param, head_dim, prefix):
    """Convert gated attention QGKV from Megatron per-group layout to HF format."""
    qg, k, v = convert_gated_attn_qgkv_mcore_to_hf(
        param,
        num_heads=args.num_attention_heads,
        num_kv_heads=args.num_query_groups,
        head_dim=head_dim,
        hidden_size=args.hidden_size,
    )
    return [
        (f"{prefix}.self_attn.q_proj.weight", qg),
        (f"{prefix}.self_attn.k_proj.weight", k),
        (f"{prefix}.self_attn.v_proj.weight", v),
    ]


def _convert_gdn_in_proj_to_hf(args, param, prefix):
    """Convert native GDN in_proj from MCore layout to HF format."""
    qkvz, ba = convert_gdn_in_proj_mcore_to_hf(
        param,
        num_k_heads=args.linear_num_key_heads,
        num_v_heads=args.linear_num_value_heads,
        key_head_dim=args.linear_key_head_dim,
        value_head_dim=args.linear_value_head_dim,
    )
    return [
        (f"{prefix}.linear_attn.in_proj_qkvz.weight", qkvz),
        (f"{prefix}.linear_attn.in_proj_ba.weight", ba),
    ]


def _convert_mtp_layer(args, name, param, layer_idx):
    """Convert MTP layer parameters from Megatron to HuggingFace format.

    Handles both wrapper layers (enorm, hnorm, final_layernorm, eh_proj) and
    inner transformer layers for any number of MTP layers.
    """
    # MTP wrapper layers (layer index independent in HF format)
    if "enorm.weight" in name:
        return [("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        return [("mtp.fc.weight", param)]

    # MTP inner transformer layers (keep layer index)
    if "transformer_layer" in name:
        proxy_name = name.replace(f"mtp.layers.{layer_idx}.transformer_layer", f"decoder.layers.{layer_idx}")
        mapped_params = convert_qwen3_next_to_hf(args, proxy_name, param)

        final_params = []
        for hf_name, tensor in mapped_params:
            target_prefix = f"mtp.layers.{layer_idx}"
            if f"model.layers.{layer_idx}" in hf_name:
                new_hf_name = hf_name.replace(f"model.layers.{layer_idx}", target_prefix)
                final_params.append((new_hf_name, tensor))
            else:
                final_params.append((hf_name, tensor))
        return final_params

    return None


def convert_qwen3_next_to_hf(args, name, param):
    """Convert Qwen3 Next model parameters from Megatron to HuggingFace format."""
    # Handle MTP layers
    if "mtp.layers" in name:
        parts = name.split(".")
        try:
            layer_idx_loc = parts.index("layers") + 1
            layer_idx = parts[layer_idx_loc]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e

        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        # --- Full attention layers ---
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            # MCore uses "linear_qkv" even with attention_output_gate=True (QGKV layout)
            assert getattr(args, "attention_output_gate", False), "attention_output_gate must be True for qwen3 next"
            return _convert_qgkv_weight_to_hf(args, param, head_dim, f"model.layers.{layer_idx}")

        # --- Native GDN (linear attention) layers ---
        elif rest == "self_attention.in_proj.weight":
            return _convert_gdn_in_proj_to_hf(args, param, f"model.layers.{layer_idx}")
        elif rest == "self_attention.in_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "self_attention.conv1d.weight":
            return [(f"model.layers.{layer_idx}.linear_attn.conv1d.weight", param)]
        elif rest == "self_attention.dt_bias":
            return [(f"model.layers.{layer_idx}.linear_attn.dt_bias", param)]
        elif rest == "self_attention.A_log":
            return [(f"model.layers.{layer_idx}.linear_attn.A_log", param)]
        elif rest == "self_attention.out_norm.weight":
            # GDN out_norm uses direct-scale weights (gdn_out_norm_zero_centered_gamma=False),
            # matching HF format directly. No +1/-1 adjustment needed.
            return [(f"model.layers.{layer_idx}.linear_attn.norm.weight", param)]
        elif rest == "self_attention.out_proj.weight":
            return [(f"model.layers.{layer_idx}.linear_attn.out_proj.weight", param)]

        # --- MLP and other layers ---
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
