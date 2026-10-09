from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.utils import MoEStateDictAdapter

from miles.utils.hf_utils.config import load_hf_config

_HF_LAYER = "model.language_model.layers.{}"

_LAYER_MAP = {
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "hc_attn_fn": "hc_attn.fn",
    "hc_attn_base": "hc_attn.base",
    "hc_attn_scale": "hc_attn.scale",
    "hc_ffn_fn": "hc_ffn.fn",
    "hc_ffn_base": "hc_ffn.base",
    "hc_ffn_scale": "hc_ffn.scale",
    **{
        f"self_attn.{name}.weight": f"attn.{name}.weight"
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_conv1d",
            "k_conv1d",
            "v_conv1d",
            "b_proj",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
            "o_norm",
        )
    },
    "self_attn.A_log": "attn.gate.A_log",
    "self_attn.dt_bias": "attn.gate.dt_bias",
    "self_attn.q_a_proj.weight": "attn.wq_a.weight",
    "self_attn.q_a_layernorm.weight": "attn.q_norm.weight",
    "self_attn.q_b_proj.weight": "attn.wq_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attn.wkv_a.weight",
    "self_attn.kv_a_layernorm.weight": "attn.kv_norm.weight",
    "self_attn.kv_b_proj.weight": "attn.wkv_b.weight",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    **{
        f"self_attn.indexer.{name}": f"attn.indexer.{name}"
        for name in (
            "wq_b.weight",
            "wk.weight",
            "k_norm.weight",
            "k_norm.bias",
            "weights_proj.weight",
            "index_kpool_compress_gate",
            "index_kpool_compress_ape",
        )
    },
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.experts.{}.gate_proj.weight": "moe.routed_experts.inner_experts.w1_EFD",
    "mlp.experts.{}.up_proj.weight": "moe.routed_experts.inner_experts.w3_EFD",
    "mlp.experts.{}.down_proj.weight": "moe.routed_experts.inner_experts.w2_EDF",
    "mlp.gate.weight": "moe.router.gate.weight",
    "mlp.gate.e_score_correction_bias": "moe.expert_bias_E",
    "mlp.shared_experts.gate_proj.weight": "moe.shared_experts.w1.weight",
    "mlp.shared_experts.up_proj.weight": "moe.shared_experts.w3.weight",
    "mlp.shared_experts.down_proj.weight": "moe.shared_experts.w2.weight",
}


class Glm5NextStateDictAdapter(DeepSeekV3StateDictAdapter):
    def __init__(self, model_config, hf_assets_path: str | None):
        MoEStateDictAdapter.__init__(self, model_config, hf_assets_path)
        if hf_assets_path is not None:
            _check_layer_layout(model_config, hf_assets_path)
        self.from_hf_map = {
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            **{f"{_HF_LAYER}.{hf}": f"layers.{{}}.{titan}" for hf, titan in _LAYER_MAP.items()},
        }

    def _validate_hf_rope_config(self, expected_rope_cls: type) -> None:
        pass


def _check_layer_layout(model_config, hf_assets_path: str) -> None:
    text_config = load_hf_config(hf_assets_path)
    text_config = getattr(text_config, "text_config", None) or text_config
    expected = [
        ("linear_attention" if layer.kda is not None else "deepseek_sparse_attention", layer.moe is not None)
        for layer in model_config.layers
    ]
    actual = [
        (layer_type, mlp_type == "sparse")
        for layer_type, mlp_type in zip(text_config.layer_types, text_config.mlp_layer_types, strict=True)
    ]
    if expected != actual:
        raise ValueError(
            f"torchtitan flavor layout {expected} does not match the checkpoint's layer_types / "
            f"mlp_layer_types {actual}"
        )
