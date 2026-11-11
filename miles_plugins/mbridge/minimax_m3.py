"""Bridge for MiniMax-M3 (HF ``MiniMaxM3SparseForConditionalGeneration``, text part only).

Text weights live under ``language_model.``; the vision tower and projector are never
created on the Megatron side.
"""

import torch
from transformers import PretrainedConfig

from mbridge.core import LLMBridge, register_model
from miles_plugins.models.minimax_m3.minimax_m3 import apply_minimax_m3_config

_LAYER = "language_model.model.layers.{layer_number}"


@register_model(["minimax_m3_vl", "minimax_m3"])
class MiniMaxM3Bridge(LLMBridge):
    _CONFIG_MAPPING = {**LLMBridge._CONFIG_MAPPING, "attention_dropout": ("attention_dropout", 0.0)}

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "language_model.model.norm.weight",
        "output_layer.weight": "language_model.lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [f"{_LAYER}.self_attn.o_proj.weight"],
        "self_attention.q_layernorm.weight": [f"{_LAYER}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": [f"{_LAYER}.self_attn.k_norm.weight"],
        "self_attention.linear_qkv.weight": [
            f"{_LAYER}.self_attn.q_proj.weight",
            f"{_LAYER}.self_attn.k_proj.weight",
            f"{_LAYER}.self_attn.v_proj.weight",
        ],
        "self_attention.indexer.index_q_proj.weight": [f"{_LAYER}.self_attn.index_q_proj.weight"],
        "self_attention.indexer.index_k_proj.weight": [f"{_LAYER}.self_attn.index_k_proj.weight"],
        "self_attention.indexer.index_q_norm": [f"{_LAYER}.self_attn.index_q_norm.weight"],
        "self_attention.indexer.index_k_norm": [f"{_LAYER}.self_attn.index_k_norm.weight"],
    }

    # first match wins, so the per-expert and shared-expert keys precede the dense ones
    _MLP_MAPPING = {
        "mlp.experts.linear_fc1": [
            f"{_LAYER}.block_sparse_moe.experts.{{expert_id}}.w1.weight",
            f"{_LAYER}.block_sparse_moe.experts.{{expert_id}}.w3.weight",
        ],
        "mlp.experts.linear_fc2": [f"{_LAYER}.block_sparse_moe.experts.{{expert_id}}.w2.weight"],
        "mlp.shared_experts.linear_fc1.weight": [
            f"{_LAYER}.block_sparse_moe.shared_experts.gate_proj.weight",
            f"{_LAYER}.block_sparse_moe.shared_experts.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": [f"{_LAYER}.block_sparse_moe.shared_experts.down_proj.weight"],
        "mlp.router.expert_bias": [f"{_LAYER}.block_sparse_moe.e_score_correction_bias"],
        "mlp.router.weight": [f"{_LAYER}.block_sparse_moe.gate.weight"],
        "pre_mlp_layernorm.weight": [f"{_LAYER}.post_attention_layernorm.weight"],
        "mlp.linear_fc1.layer_norm_weight": [f"{_LAYER}.post_attention_layernorm.weight"],
        "mlp.linear_fc1.weight": [f"{_LAYER}.mlp.gate_proj.weight", f"{_LAYER}.mlp.up_proj.weight"],
        "mlp.linear_fc2.weight": [f"{_LAYER}.mlp.down_proj.weight"],
    }

    _OTHER_MAPPING = {
        "input_layernorm.weight": [f"{_LAYER}.input_layernorm.weight"],
    }

    def __init__(self, hf_config, *args, **kwargs):
        text_config = getattr(hf_config, "text_config", None) or hf_config  # config-access-exempt: optional key
        # the remote config class keeps the nested text config as a plain dict
        if isinstance(text_config, dict):
            text_config = PretrainedConfig.from_dict(text_config)
        super().__init__(text_config, *args, **kwargs)

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        for keyword, targets in self._MLP_MAPPING.items():
            if keyword in name:
                expert_id = name.split("weight")[-1] if "{expert_id}" in targets[0] else None
                return [t.format(layer_number=layer_number, expert_id=expert_id) for t in targets]
        raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        if mcore_weights_name.endswith("mlp.router.expert_bias"):
            return hf_weights[0].to(torch.float32)
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _build_config(self):
        cfg = self.hf_config
        config = self._build_base_config(
            use_cpu_initialization=False,
            num_moe_experts=cfg.num_local_experts,
            moe_ffn_hidden_size=cfg.intermediate_size,
            ffn_hidden_size=cfg.dense_intermediate_size,
            moe_shared_expert_intermediate_size=cfg.shared_intermediate_size * cfg.n_shared_experts,
            moe_router_topk=cfg.num_experts_per_tok,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0.0,
            moe_router_topk_scaling_factor=cfg.routed_scaling_factor,
            moe_router_load_balancing_type="none",
            moe_router_dtype="fp32",
            moe_grouped_gemm=True,
            moe_layer_freq=list(cfg.moe_layer_freq),
            qk_layernorm=True,
            layernorm_zero_centered_gamma=True,
            add_qkv_bias=False,
            add_bias_linear=False,
            kv_channels=cfg.head_dim,
            rotary_interleaved=False,
        )
        apply_minimax_m3_config(config, cfg)
        config.activation_func_clamp_value = cfg.swiglu_limit
        return config
