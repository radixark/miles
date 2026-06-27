"""mbridge weight bridge for MiniMax-M3 (used by tools/convert_hf_to_torch_dist.py).

The offline HF->torch_dist converter uses the ``mbridge`` package (not
``megatron.bridge``). The Megatron model itself is built via ``--spec
get_minimax_m3_spec`` (so MSA attention is present); mbridge only supplies the
HF<->Megatron parameter NAME mappings + the provider config.

Mappings verified against the real checkpoint key layout
(``model.safetensors.index.json``), prefix ``language_model.model.*``:

    language_model.model.layers.{i}.input_layernorm.weight
    language_model.model.layers.{i}.post_attention_layernorm.weight
    language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    language_model.model.layers.{i}.self_attn.{q,k}_norm.weight
    language_model.model.layers.{i}.self_attn.index_{q,k}_{proj,norm}.weight   # sparse layers
    language_model.model.layers.{i}.mlp.{gate,up,down}_proj.weight             # dense layers 0-2
    language_model.model.layers.{i}.block_sparse_moe.gate.weight               # MoE layers 3-59
    language_model.model.layers.{i}.block_sparse_moe.e_score_correction_bias
    language_model.model.layers.{i}.block_sparse_moe.experts.{e}.{w1,w3,w2}.weight
    language_model.model.layers.{i}.block_sparse_moe.shared_experts.{gate,up,down}_proj.weight

NOTE: structurally complete and key-verified, but the offline 428B conversion run
is the validation step (mbridge mixed dense/MoE handling + the indexer params).
"""

import torch

from mbridge.core import register_model
from mbridge.models import Qwen3MoEBridge


_P = "language_model.model"


@register_model(["minimax_m3", "minimax_m3_vl"])
class MiniMaxM3Bridge(Qwen3MoEBridge):
    """MiniMax-M3: GQA + per-head QK-norm + MSA indexer + DeepSeek-style MoE."""

    def __init__(self, hf_config, *args, **kwargs):
        # M3 is a VLM-nested config; the LM fields the base reads
        # (num_hidden_layers, hidden_size, head_dim, ...) live under text_config.
        # Flatten to text_config so base config resolution + _CONFIG_MAPPING work.
        # Weight keys (language_model.model.*) are unaffected. Keep the original
        # for any top-level VL fields.
        self._full_hf_config = hf_config
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        super().__init__(hf_config, *args, **kwargs)

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": f"{_P}.embed_tokens.weight",
        "decoder.final_layernorm.weight": f"{_P}.norm.weight",
        "output_layer.weight": "language_model.lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [f"{_P}.layers.{{layer_number}}.self_attn.o_proj.weight"],
        "self_attention.linear_qkv.layer_norm_weight": [f"{_P}.layers.{{layer_number}}.input_layernorm.weight"],
        "self_attention.q_layernorm.weight": [f"{_P}.layers.{{layer_number}}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": [f"{_P}.layers.{{layer_number}}.self_attn.k_norm.weight"],
        "self_attention.linear_qkv.weight": [
            f"{_P}.layers.{{layer_number}}.self_attn.q_proj.weight",
            f"{_P}.layers.{{layer_number}}.self_attn.k_proj.weight",
            f"{_P}.layers.{{layer_number}}.self_attn.v_proj.weight",
        ],
        # MSA lightning indexer (sparse layers only). Param names match
        # MSASelfAttentionSubmodules attribute names in minimax_m3.py.
        "self_attention.index_q_proj.weight": [f"{_P}.layers.{{layer_number}}.self_attn.index_q_proj.weight"],
        "self_attention.index_k_proj.weight": [f"{_P}.layers.{{layer_number}}.self_attn.index_k_proj.weight"],
        "self_attention.index_q_norm.weight": [f"{_P}.layers.{{layer_number}}.self_attn.index_q_norm.weight"],
        "self_attention.index_k_norm.weight": [f"{_P}.layers.{{layer_number}}.self_attn.index_k_norm.weight"],
    }

    _MLP_MAPPING = {
        # dense layers (0..first_dense-1): plain gated MLP
        "mlp.linear_fc1.weight": [
            f"{_P}.layers.{{layer_number}}.mlp.gate_proj.weight",
            f"{_P}.layers.{{layer_number}}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [f"{_P}.layers.{{layer_number}}.post_attention_layernorm.weight"],
        "mlp.linear_fc2.weight": [f"{_P}.layers.{{layer_number}}.mlp.down_proj.weight"],
        # MoE layers: router (+ expert bias), per-expert w1/w3/w2, shared expert
        "pre_mlp_layernorm.weight": [f"{_P}.layers.{{layer_number}}.post_attention_layernorm.weight"],
        "mlp.router.weight": [f"{_P}.layers.{{layer_number}}.block_sparse_moe.gate.weight"],
        "mlp.router.expert_bias": [f"{_P}.layers.{{layer_number}}.block_sparse_moe.e_score_correction_bias"],
        "mlp.experts.linear_fc1": [
            f"{_P}.layers.{{layer_number}}.block_sparse_moe.experts.{{expert_id}}.w1.weight",
            f"{_P}.layers.{{layer_number}}.block_sparse_moe.experts.{{expert_id}}.w3.weight",
        ],
        "mlp.experts.linear_fc2": [f"{_P}.layers.{{layer_number}}.block_sparse_moe.experts.{{expert_id}}.w2.weight"],
        "mlp.shared_experts.linear_fc1.weight": [
            f"{_P}.layers.{{layer_number}}.block_sparse_moe.shared_experts.gate_proj.weight",
            f"{_P}.layers.{{layer_number}}.block_sparse_moe.shared_experts.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": [
            f"{_P}.layers.{{layer_number}}.block_sparse_moe.shared_experts.down_proj.weight"
        ],
    }

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "dense_intermediate_size",   # dense layers use the big FFN
        "attention_dropout": ("attention_dropout", 0.0),
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    def _get_text_config(self):
        return getattr(self.hf_config, "text_config", self.hf_config)

    def _build_config(self):
        tc = self._get_text_config()
        sac = getattr(tc, "sparse_attention_config", {}) or {}
        sg = (lambda k, d=None: sac.get(k, d) if isinstance(sac, dict) else getattr(sac, k, d))
        cfg = self._build_base_config(
            use_cpu_initialization=False,
            # MoE — DeepSeek-V3-style sigmoid routing + expert bias correction
            moe_ffn_hidden_size=tc.intermediate_size,            # expert FFN (3072)
            moe_router_topk=tc.num_experts_per_tok,
            num_moe_experts=tc.num_local_experts,
            moe_shared_expert_intermediate_size=tc.n_shared_experts * tc.intermediate_size,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_pre_softmax=False,
            moe_router_topk_scaling_factor=getattr(tc, "routed_scaling_factor", 1.0),
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            moe_aux_loss_coeff=0.0,
            num_layers_in_first_pipeline_stage=None,
            qk_layernorm=True,
        )
        # SwiGLU-OAI activation (== GPT-OSS): quick_gelu gate + clamp + (up+1)
        if str(getattr(tc, "hidden_act", "")).lower() in ("swigluoai", "swiglu_oai"):
            from megatron.core.activations import quick_gelu

            cfg.activation_func = quick_gelu
            cfg.activation_func_clamp_value = float(getattr(tc, "swiglu_limit", 7.0))
            cfg.glu_linear_offset = 1.0
            cfg.gated_linear_unit = True
            cfg.bias_activation_fusion = False  # non-fused glu handles clamp+offset
        if getattr(tc, "use_gemma_norm", False):
            cfg.layernorm_zero_centered_gamma = True
        # MSA hyperparameters (consumed by get_minimax_m3_spec / MSASelfAttention)
        cfg.sparse_index_dim = sg("sparse_index_dim")
        cfg.sparse_num_index_heads = sg("sparse_num_index_heads")
        cfg.sparse_block_size = sg("sparse_block_size")
        cfg.sparse_topk_blocks = sg("sparse_topk_blocks")
        cfg.sparse_init_block = sg("sparse_init_block", 0)
        cfg.sparse_local_block = sg("sparse_local_block", 1)
        # rope (rotary_base / rotary_percent) comes from the --rotary-base /
        # --rotary-percent CLI args; not a TransformerConfig field, don't set here.
        return cfg
