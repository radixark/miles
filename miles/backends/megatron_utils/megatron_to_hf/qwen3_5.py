import functools
import re

import torch
from transformers import AutoConfig


@functools.lru_cache(maxsize=4)
def _get_gdn_dims(hf_checkpoint: str):
    """Return (qk_dim, v_dim, num_v_heads) for the GatedDeltaNet in_proj split."""
    cfg = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    qk_dim = tc.linear_num_key_heads * tc.linear_key_head_dim
    v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim
    num_v_heads = tc.linear_num_value_heads
    return qk_dim, v_dim, num_v_heads


def _convert_mtp_layer(args, name, param, layer_idx):
    """Convert MTP layer parameters from Megatron to HuggingFace format."""
    if "enorm.weight" in name:
        return [("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        return [("mtp.fc.weight", param)]

    if "transformer_layer" in name:
        proxy_name = name.replace(f"mtp.layers.{layer_idx}.transformer_layer", f"decoder.layers.{layer_idx}")
        mapped_params = convert_qwen3_5_to_hf(args, proxy_name, param)

        final_params = []
        for hf_name, tensor in mapped_params:
            target_prefix = f"mtp.layers.{layer_idx}"
            if f"model.language_model.layers.{layer_idx}" in hf_name:
                new_hf_name = hf_name.replace(f"model.language_model.layers.{layer_idx}", target_prefix)
                final_params.append((new_hf_name, tensor))
            else:
                final_params.append((hf_name, tensor))
        return final_params

    return None


def convert_qwen3_5_to_hf(args, name, param):
    """Convert Qwen3.5 model parameters from Megatron to HuggingFace format.

    Qwen3.5 uses model.language_model.layers prefix and has separate
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a for linear attention.
    """
    # Vision encoder weights exist in the Megatron model when the HF config is a VLM
    # (has vision_config), but are not needed for text-only training or SGLang inference.
    if "vision_model" in name:
        return []

    # VLM configs nest the text model under language_model — normalize so all
    # handlers below work regardless of whether the model is pure-text or VLM.
    name = name.replace("module.module.language_model.", "module.module.", 1)

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
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        prefix = f"model.language_model.layers.{layer_idx}"

        # experts (grouped gemm - fused format)
        if rest == "mlp.experts.linear_fc1":
            return [(f"{prefix}.mlp.experts.gate_up_proj", param)]
        elif rest == "mlp.experts.linear_fc2":
            return [(f"{prefix}.mlp.experts.down_proj", param)]

        # experts (ungrouped - individual expert format)
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight", param)]
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
                    (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"{prefix}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"{prefix}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"{prefix}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
            )
            q_param = (
                q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
                .transpose(1, 2)
                .reshape(-1, args.hidden_size)
            )
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{prefix}.self_attn.q_proj.weight", q_param),
                (f"{prefix}.self_attn.k_proj.weight", k_param),
                (f"{prefix}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"{prefix}.self_attn.q_proj.bias", q_bias),
                (f"{prefix}.self_attn.k_proj.bias", k_bias),
                (f"{prefix}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.gate_proj.weight", gate_weight),
                (f"{prefix}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"{prefix}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"{prefix}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"{prefix}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{prefix}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{prefix}.self_attn.k_norm.weight", param)]
        elif rest.startswith("self_attention.") and rest[len("self_attention.") :] in [
            "input_layernorm.weight",
            # linear attn (Qwen3.5 uses separate in_proj_b/in_proj_a)
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn (full attention layers)
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]:
            rest = rest[len("self_attention.") :]
            return [(f"{prefix}.{rest}", param)]

        # Fused input layernorm from GatedDeltaNet's ColumnParallelLinearWithNorm.
        elif rest == "self_attention.in_proj.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", param)]

        # Combined in_proj from native GatedDeltaNet — split into the 4 HF projections.
        elif rest == "self_attention.in_proj.weight":
            qk_dim, v_dim, num_v_heads = _get_gdn_dims(args.hf_checkpoint)
            in_proj_qkv, in_proj_z, in_proj_b, in_proj_a = param.split(
                [qk_dim * 2 + v_dim, v_dim, num_v_heads, num_v_heads], dim=0
            )
            return [
                (f"{prefix}.linear_attn.in_proj_qkv.weight", in_proj_qkv),
                (f"{prefix}.linear_attn.in_proj_z.weight", in_proj_z),
                (f"{prefix}.linear_attn.in_proj_b.weight", in_proj_b),
                (f"{prefix}.linear_attn.in_proj_a.weight", in_proj_a),
            ]

        # Direct SSM params when GatedDeltaNet sits at self_attention level without a
        # linear_attn wrapper — map to the linear_attn.* names HF expects.
        # out_norm → norm: Megatron calls it out_norm, HF calls it norm.
        elif rest.startswith("self_attention."):
            _SSM_RENAME = {
                "dt_bias": "dt_bias",
                "A_log": "A_log",
                "conv1d.weight": "conv1d.weight",
                "out_norm.weight": "norm.weight",
                "out_proj.weight": "out_proj.weight",
                "in_proj_qkv.weight": "in_proj_qkv.weight",
                "in_proj_z.weight": "in_proj_z.weight",
                "in_proj_b.weight": "in_proj_b.weight",
                "in_proj_a.weight": "in_proj_a.weight",
            }
            sub_name = rest[len("self_attention."):]
            if sub_name in _SSM_RENAME:
                return [(f"{prefix}.linear_attn.{_SSM_RENAME[sub_name]}", param)]

    raise ValueError(f"Unknown parameter name: {name}")
