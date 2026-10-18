"""Load Step5 HF weights into Megatron through mbridge.

Virtual per-expert names read and dequantize rows of packed FP8 tensors. QKV/gate
weights are arranged in TP-rank blocks for mbridge's even split. Norms, router bias,
and FP32 indexer parameters retain their checkpoint dtype; exports use Miles'
`megatron_to_hf` converter.
"""

import os
import re

import torch
from safetensors import safe_open

from mbridge.core import LLMBridge, register_model
from mbridge.core.safetensor_io import SafeTensorIO
from miles_plugins.models.step5.indexer import CHECKPOINT_NAMES as INDEXER_CHECKPOINT_NAMES
from miles_plugins.models.step5.indexer import FP32_PARAMS as INDEXER_FP32_PARAMS
from miles_plugins.models.step5.qkv_layout import fuse_qkv_and_gate

_LAYER = "model.layers.{layer_number}"
_VIRTUAL_EXPERT = re.compile(r"^(model\.layers\.\d+\.moe)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_FP8_BLOCK = 128
_FP32_WEIGHTS = (
    "layernorm.weight",  # input_layernorm, pre_mlp_layernorm, q_layernorm, k_layernorm, final_layernorm
    "mlp.router.expert_bias",
    *(f"self_attention.indexer.{name}" for name in INDEXER_FP32_PARAMS),
)


@register_model(["step5_vl", "step5"])
class Step5Bridge(LLMBridge):
    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_attention_groups",
        "ffn_hidden_size": "intermediate_size",
        "layernorm_epsilon": "rms_norm_eps",
        "kv_channels": "head_dim",
    }

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_qkv.weight": [
            f"{_LAYER}.self_attn.q_proj.weight",
            f"{_LAYER}.self_attn.k_proj.weight",
            f"{_LAYER}.self_attn.v_proj.weight",
            f"{_LAYER}.self_attn.g_proj.weight",
        ],
        "self_attention.q_layernorm.weight": [f"{_LAYER}.self_attn.q_norm.weight"],
        "self_attention.k_layernorm.weight": [f"{_LAYER}.self_attn.k_norm.weight"],
        "self_attention.linear_proj.weight": [f"{_LAYER}.self_attn.o_proj.weight"],
        **{
            f"self_attention.indexer.{name}": [f"{_LAYER}.self_attn.{checkpoint_name}"]
            for name, checkpoint_name in INDEXER_CHECKPOINT_NAMES.items()
        },
    }

    _MLP_MAPPING = {
        "pre_mlp_layernorm.weight": [f"{_LAYER}.post_attention_layernorm.weight"],
        "mlp.linear_fc1.weight": [f"{_LAYER}.mlp.gate_proj.weight", f"{_LAYER}.mlp.up_proj.weight"],
        "mlp.linear_fc2.weight": [f"{_LAYER}.mlp.down_proj.weight"],
        "mlp.router.weight": [f"{_LAYER}.moe.gate.weight"],
        "mlp.router.expert_bias": [f"{_LAYER}.moe.router_bias"],
        "mlp.shared_experts.linear_fc1.weight": [
            f"{_LAYER}.share_expert.gate_proj.weight",
            f"{_LAYER}.share_expert.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": [f"{_LAYER}.share_expert.down_proj.weight"],
        "mlp.experts.linear_fc1.weight": [
            f"{_LAYER}.moe.experts.{{expert_id}}.gate_proj.weight",
            f"{_LAYER}.moe.experts.{{expert_id}}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2.weight": [f"{_LAYER}.moe.experts.{{expert_id}}.down_proj.weight"],
    }

    _OTHER_MAPPING = {
        "input_layernorm.weight": [f"{_LAYER}.input_layernorm.weight"],
    }

    @property
    def text_config(self):
        return getattr(self.hf_config, "text_config", self.hf_config)

    def _build_config(self):
        # The same model scripts/models/step5.py describes on the command line, so a config built
        # here alone (bridge.get_model) routes and windows attention as the checkpoint does.
        text_config = self.text_config
        moe_layers = set(text_config.moe_layer_list)
        layer_types = text_config.layer_types[: text_config.num_hidden_layers]
        return self._build_base_config(
            text_config_key="text_config" if self.text_config is not self.hf_config else None,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            qk_layernorm=True,
            head_wise_attn_gate=True,
            fp32_residual_connection=True,
            num_moe_experts=text_config.moe_num_experts,
            moe_ffn_hidden_size=text_config.moe_intermediate_size,
            moe_shared_expert_intermediate_size=text_config.share_expert_dim,
            moe_router_topk=text_config.moe_top_k,
            moe_layer_freq=[int(index in moe_layers) for index in range(text_config.num_hidden_layers)],
            moe_grouped_gemm=True,
            moe_router_dtype="fp32",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0.0,
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_router_topk_scaling_factor=text_config.moe_router_scaling_factor,
            moe_combine_in_fp32=True,
            # a 512-token window is the query plus 511 before it; a 1 in the skip list selects the window
            window_size=(text_config.sliding_window - 1, 0),
            window_attn_skip_freq=[int(kind == "sliding_attention") for kind in layer_types],
        )

    def _get_gptmodel_args(self) -> dict:
        return {
            "vocab_size": self.text_config.vocab_size,
            "max_sequence_length": self.text_config.max_position_embeddings,
            "position_embedding_type": "rope",
        }

    def _get_safetensor_io(self, weights_path: str):
        return Step5SafeTensorIO(self._get_actual_hf_path(weights_path))

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        # Grouped experts are named ``...linear_fc1.weight<global expert id>``.
        expert_id = name.rsplit(".weight", 1)[1] if ".mlp.experts." in name else None
        for keyword, hf_names in self._MLP_MAPPING.items():
            if keyword in name:
                return [hf_name.format(layer_number=layer_number, expert_id=expert_id) for hf_name in hf_names]
        raise NotImplementedError(f"Unsupported parameter name: {name}")

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        if mcore_weights_name.endswith(_FP32_WEIGHTS):
            (weight,) = hf_weights
            return weight.float()
        if mcore_weights_name.endswith("self_attention.linear_qkv.weight"):
            query, key, value, gate = (weight.to(self.dtype) for weight in hf_weights)
            return fuse_qkv_and_gate(
                query=query,
                key=key,
                value=value,
                gate=gate,
                num_query_groups=self.config.num_query_groups,
                tp_size=self.mpu.tp_size,
            )
        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(self, mcore_weights_name: str, mcore_weights: torch.Tensor):
        raise NotImplementedError("Step5 weights export through miles' megatron_to_hf converter")


def dequantize_block_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """``weight * scale_inv[block]`` for one ``[out, in]`` expert with 128x128 blocks."""
    rows, cols = weight.shape
    scale = scale_inv.repeat_interleave(_FP8_BLOCK, dim=0)[:rows].repeat_interleave(_FP8_BLOCK, dim=1)[:, :cols]
    return (weight.float() * scale.float()).to(torch.bfloat16)


class Step5SafeTensorIO(SafeTensorIO):
    """Serves virtual per-expert names from Step5's packed expert tensors."""

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict:
        experts = [name for name in hf_weight_names if _VIRTUAL_EXPERT.match(name)]
        stored = [name for name in hf_weight_names if not _VIRTUAL_EXPERT.match(name)]
        loaded = super().load_some_hf_weight(stored) if stored else {}
        return loaded | {name: self._load_expert(name) for name in experts}

    def _load_expert(self, name: str) -> torch.Tensor:
        prefix, expert, projection = _VIRTUAL_EXPERT.match(name).groups()
        packed = f"{prefix}.{projection}.weight"
        weight = self._read_row(packed, int(expert))
        scale_name = f"{packed}_scale_inv"
        if scale_name not in self.index:
            return weight
        return dequantize_block_fp8(weight, self._read_row(scale_name, int(expert)))

    def _read_row(self, name: str, row: int) -> torch.Tensor:
        with safe_open(os.path.join(self.hf_dir, self.index[name]), framework="pt", device="cpu") as f:
            return f.get_slice(name)[row : row + 1][0]
