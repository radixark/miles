from argparse import Namespace

import torch

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


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


def convert_gated_attn_qgkv_mcore_to_hf(mcore_weights, *, num_heads, num_kv_heads, head_dim, hidden_size):
    """Convert MCore per-group [Q_g, G_g, K_g, V_g] → HF [q_proj(Q+G interleaved), k_proj, v_proj]."""
    heads_per_group = num_heads // num_kv_heads
    q_size = heads_per_group * head_dim
    g_size = heads_per_group * head_dim
    group_size = q_size + g_size + head_dim + head_dim

    groups = mcore_weights.view(num_kv_heads, group_size, hidden_size)

    all_q = groups[:, :q_size, :].reshape(num_kv_heads, heads_per_group, head_dim, hidden_size)
    all_g = groups[:, q_size : q_size + g_size, :].reshape(num_kv_heads, heads_per_group, head_dim, hidden_size)
    all_k = groups[:, q_size + g_size : q_size + g_size + head_dim, :]
    all_v = groups[:, q_size + g_size + head_dim :, :]

    q = all_q.reshape(num_heads, head_dim, hidden_size)
    g = all_g.reshape(num_heads, head_dim, hidden_size)
    qg = torch.cat([q, g], dim=1).reshape(num_heads * 2 * head_dim, hidden_size).contiguous()

    k = all_k.reshape(num_kv_heads * head_dim, hidden_size).contiguous()
    v = all_v.reshape(num_kv_heads * head_dim, hidden_size).contiguous()

    return [qg, k, v]


def convert_gdn_in_proj_mcore_to_hf(mcore_weights, *, num_k_heads, num_v_heads, key_head_dim, value_head_dim):
    """Convert MCore [Q, K, V, Z, beta, alpha] → HF [in_proj_qkvz, in_proj_ba]."""
    nv_per = num_v_heads // num_k_heads
    qk_dim = num_k_heads * key_head_dim
    v_dim = num_v_heads * value_head_dim
    hidden_size = mcore_weights.shape[1]

    Q, K, V, Z, beta, alpha = torch.split(
        mcore_weights, [qk_dim, qk_dim, v_dim, v_dim, num_v_heads, num_v_heads], dim=0
    )

    Q = Q.view(num_k_heads, key_head_dim, hidden_size)
    K = K.view(num_k_heads, key_head_dim, hidden_size)
    V = V.view(num_k_heads, nv_per * value_head_dim, hidden_size)
    Z = Z.view(num_k_heads, nv_per * value_head_dim, hidden_size)
    qkvz = torch.cat([Q, K, V, Z], dim=1).reshape(-1, hidden_size).contiguous()

    beta = beta.view(num_k_heads, nv_per, hidden_size)
    alpha = alpha.view(num_k_heads, nv_per, hidden_size)
    ba = torch.cat([beta, alpha], dim=1).reshape(-1, hidden_size).contiguous()

    return [qkvz, ba]


@register_model("qwen3_next")
class Qwen3NextBridge(Qwen2MoEBridge):
    _ATTENTION_MAPPING = (
        Qwen2MoEBridge._ATTENTION_MAPPING
        # Native GDN (linear attention) weight mappings
        | {
            # in_proj fuses layernorm + combined [Q, K, V, Z, beta, alpha] projection
            "self_attention.in_proj.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
            "self_attention.in_proj.weight": [
                "model.layers.{layer_number}.linear_attn.in_proj_qkvz.weight",
                "model.layers.{layer_number}.linear_attn.in_proj_ba.weight",
            ],
            # 1:1 mappings for other GDN weights
            "self_attention.conv1d.weight": ["model.layers.{layer_number}.linear_attn.conv1d.weight"],
            "self_attention.dt_bias": ["model.layers.{layer_number}.linear_attn.dt_bias"],
            "self_attention.A_log": ["model.layers.{layer_number}.linear_attn.A_log"],
            "self_attention.out_norm.weight": ["model.layers.{layer_number}.linear_attn.norm.weight"],
            "self_attention.out_proj.weight": ["model.layers.{layer_number}.linear_attn.out_proj.weight"],
        }
        # NOTE: Gated attention (full attention layers with attention_output_gate=True) still uses
        # the MCore attribute name "linear_qkv" (not "linear_qgkv") — the parent's mapping for
        # "self_attention.linear_qkv.weight" → [q_proj, k_proj, v_proj] is correct.
        # The gated Q+G ↔ separate Q,G conversion is handled in _weight_to_mcore_format / _weight_to_hf_format.
    )

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """Override to handle MTP layer mappings."""
        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _convert_mtp_param(self, name: str) -> list[str]:
        """Convert MTP layer parameters from MCore to HF format."""
        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        parts = name.split(".")
        mtp_layer_idx = parts[2]  # mtp.layers.{idx}

        direct_name_mapping = {
            f"mtp.layers.{mtp_layer_idx}.eh_proj.weight": "mtp.fc.weight",
            f"mtp.layers.{mtp_layer_idx}.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            f"mtp.layers.{mtp_layer_idx}.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            f"mtp.layers.{mtp_layer_idx}.final_layernorm.weight": "mtp.norm.weight",
        }

        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        if "transformer_layer" in name:
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.transformer_layer",
                f"decoder.layers.{mtp_layer_idx}",
            )

            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                convert_names = super()._weight_name_mapping_attention(proxy_name)
            elif "mlp" in proxy_name or "pre_mlp_layernorm" in proxy_name:
                convert_names = super()._weight_name_mapping_mlp(proxy_name)
            else:
                raise NotImplementedError(f"Unsupported transformer component in MTP: {name}")

            convert_names = [
                cn.replace(f"model.layers.{mtp_layer_idx}", f"mtp.layers.{mtp_layer_idx}") for cn in convert_names
            ]
            return convert_names

        raise NotImplementedError(f"Unsupported MTP parameter name: {name}")

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        if "self_attention.in_proj." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            return self._convert_gdn_in_proj_hf_to_mcore(hf_weights)

        if "self_attention.out_norm.weight" in mcore_weights_name:
            assert len(hf_weights) == 1
            out_norm_weight = hf_weights[0]
            # Qwen3-Next uses mixed norm semantics:
            # - most RMSNorms are Gemma-style (+1)
            # - linear_attn.norm is direct-scale (no need +1)
            # If Megatron enables global zero-centered gamma, compensate here so
            # GDN out_norm still computes with direct-scale weight.
            if getattr(self.config, "layernorm_zero_centered_gamma", False):
                return out_norm_weight - out_norm_weight.new_tensor(1.0)
            return out_norm_weight

        # Gated attention: MCore still names the attribute "linear_qkv" even with attention_output_gate=True,
        # but the weight contains [Q, G, K, V] per group instead of [Q, K, V].
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            return self._convert_gated_attn_qgkv_hf_to_mcore(hf_weights)

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        if "self_attention.in_proj." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            hf_weights = self._convert_gdn_in_proj_mcore_to_hf(mcore_weights)
            return hf_names, hf_weights

        if "self_attention.out_norm.weight" in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            out_norm_weight = mcore_weights
            if getattr(self.config, "layernorm_zero_centered_gamma", False):
                out_norm_weight = out_norm_weight + out_norm_weight.new_tensor(1.0)
            return hf_names, [out_norm_weight]

        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
            hf_weights = self._convert_gated_attn_qgkv_mcore_to_hf(mcore_weights)
            return hf_names, hf_weights

        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    # ---- GDN in_proj conversion: MCore [Q, K, V, Z, beta, alpha] <-> HF [qkvz, ba] ----

    def _gdn_dims(self):
        """Return GDN dimension parameters from HF config."""
        num_k_heads = self.hf_config.linear_num_key_heads
        num_v_heads = self.hf_config.linear_num_value_heads
        key_head_dim = self.hf_config.linear_key_head_dim
        value_head_dim = self.hf_config.linear_value_head_dim
        nv_per = num_v_heads // num_k_heads
        qk_dim = num_k_heads * key_head_dim
        v_dim = num_v_heads * value_head_dim
        return num_k_heads, num_v_heads, key_head_dim, value_head_dim, nv_per, qk_dim, v_dim

    def _convert_gdn_in_proj_hf_to_mcore(self, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        """Convert HF [in_proj_qkvz, in_proj_ba] → MCore [Q, K, V, Z, beta, alpha].

        HF in_proj_qkvz layout (per key group): [Q_g, K_g, V_g, Z_g]
        HF in_proj_ba layout (per key group): [beta_g, alpha_g]
        MCore in_proj layout (type-first): [Q_all, K_all, V_all, Z_all, beta_all, alpha_all]
        """
        assert len(hf_weights) == 2
        qkvz, ba = hf_weights
        num_k_heads, num_v_heads, key_hd, value_hd, nv_per, qk_dim, v_dim = self._gdn_dims()
        hidden_size = qkvz.shape[1]

        group_qkvz_dim = key_hd + key_hd + nv_per * value_hd + nv_per * value_hd
        qkvz = qkvz.view(num_k_heads, group_qkvz_dim, hidden_size)
        Q, K, V, Z = torch.split(qkvz, [key_hd, key_hd, nv_per * value_hd, nv_per * value_hd], dim=1)

        ba = ba.view(num_k_heads, 2 * nv_per, hidden_size)
        beta, alpha = torch.split(ba, [nv_per, nv_per], dim=1)

        return torch.cat(
            [
                Q.reshape(qk_dim, hidden_size),
                K.reshape(qk_dim, hidden_size),
                V.reshape(v_dim, hidden_size),
                Z.reshape(v_dim, hidden_size),
                beta.reshape(num_v_heads, hidden_size),
                alpha.reshape(num_v_heads, hidden_size),
            ],
            dim=0,
        ).contiguous()

    def _convert_gdn_in_proj_mcore_to_hf(self, mcore_weights: torch.Tensor) -> list[torch.Tensor]:
        """Convert MCore [Q, K, V, Z, beta, alpha] → HF [in_proj_qkvz, in_proj_ba]."""
        return convert_gdn_in_proj_mcore_to_hf(
            mcore_weights,
            num_k_heads=self.hf_config.linear_num_key_heads,
            num_v_heads=self.hf_config.linear_num_value_heads,
            key_head_dim=self.hf_config.linear_key_head_dim,
            value_head_dim=self.hf_config.linear_value_head_dim,
        )

    # ---- Gated attention QGKV conversion (full attention layers) ----

    def _convert_gated_attn_qgkv_hf_to_mcore(self, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        """Convert HF [q_proj(Q+G interleaved), k_proj, v_proj] → MCore per-group [Q_g, G_g, K_g, V_g]."""
        assert len(hf_weights) == 3
        qg, k, v = hf_weights

        num_heads = self.hf_config.num_attention_heads
        num_kv_heads = self.hf_config.num_key_value_heads
        head_dim = self.hf_config.head_dim
        hidden_size = self.hf_config.hidden_size
        heads_per_group = num_heads // num_kv_heads

        qg = qg.view(num_heads, 2 * head_dim, hidden_size)
        q = qg[:, :head_dim, :]
        g = qg[:, head_dim:, :]

        k = k.view(num_kv_heads, head_dim, hidden_size)
        v = v.view(num_kv_heads, head_dim, hidden_size)

        q = q.view(num_kv_heads, heads_per_group, head_dim, hidden_size)
        g = g.view(num_kv_heads, heads_per_group, head_dim, hidden_size)

        groups = []
        for i in range(num_kv_heads):
            q_g = q[i].reshape(heads_per_group * head_dim, hidden_size)
            g_g = g[i].reshape(heads_per_group * head_dim, hidden_size)
            groups.append(torch.cat([q_g, g_g, k[i], v[i]], dim=0))

        return torch.cat(groups, dim=0).contiguous()

    def _convert_gated_attn_qgkv_mcore_to_hf(self, mcore_weights: torch.Tensor) -> list[torch.Tensor]:
        """Convert MCore per-group [Q_g, G_g, K_g, V_g] → HF [q_proj(Q+G interleaved), k_proj, v_proj]."""
        return convert_gated_attn_qgkv_mcore_to_hf(
            mcore_weights,
            num_heads=self.hf_config.num_attention_heads,
            num_kv_heads=self.hf_config.num_key_value_heads,
            head_dim=self.hf_config.head_dim,
            hidden_size=self.hf_config.hidden_size,
        )

    # ---- Config ----
    # NOTE: _build_config is NOT used for model creation in raw mode (CLI args control that).
    # It is kept because self.config is read during load_weights for:
    #   - self.config.num_moe_experts (EP weight mapping)
    #   - self.config.gated_linear_unit (FC1 weight split)

    def _build_config(self):
        return self._build_base_config(
            num_moe_experts=self.hf_config.num_experts,
        )
