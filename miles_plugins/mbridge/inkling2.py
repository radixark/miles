"""HF -> Megatron loading for Inkling2. Names mirror the Megatron -> HF converter
(miles.backends.megatron_utils.megatron_to_hf.inkling2). Quantized experts (NVFP4 / MXFP8)
are dequantized to bf16, since the trainer computes in bf16."""

import re
from functools import cached_property

import torch
from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge

from miles.backends.megatron_utils.megatron_to_hf.inkling2 import (
    GLOBAL_PARAM_NAMES,
    LAYER_PARAM_NAMES,
    LINEAR_Q_PARTS,
    inkling2_quantization_config,
    linear_q_split,
)

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_EXPERT_PARAM = re.compile(r"^decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc(1|2)\.weight(\d+)$")
_LAYER_PARAM = re.compile(r"^decoder\.layers\.(\d+)\.(.+)$")


def dequantize_nvfp4(packed, scales, amax, denominator) -> torch.Tensor:
    """W = e2m1 * block_scale * amax / outer_scale_denominator, for one expert."""
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    values = torch.tensor(_E2M1_VALUES, device=packed.device)[codes]
    values = values.view(*scales.shape, 16) * scales.float().unsqueeze(-1)
    return (values.flatten(-2) * (amax.float() / denominator.float())).to(torch.bfloat16)


def dequantize_mxfp8(weight, scales) -> torch.Tensor:
    """W = fp8 * 2 ** (ue8m0 - 127) per block of 32 along the input dimension."""
    exponent = torch.exp2(scales.to(torch.int32).float() - 127)
    return (weight.float().view(*scales.shape, 32) * exponent.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


# older checkpoints declare inkling2_mm_model
@register_model(["inkling2", "inkling2_mm_model"])
class Inkling2Bridge(Qwen2MoEBridge):
    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "intermediate_size",
        "attention_dropout": ("attention_dropout", 0.0),
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    def _get_text_config(self):
        return self.hf_config.text_config

    @cached_property
    def _expert_formats(self) -> dict[str, str]:
        return inkling2_quantization_config(self.hf_config.name_or_path)["expert_formats"]

    def _expert_hf_names(self, layer: int, fc: str, expert: int) -> list[str]:
        leaf = f"model.llm.layers.{layer}.mlp.experts.{'w13' if fc == '1' else 'w2'}_weight"
        fmt = self._expert_formats[leaf]
        if fmt == "nvfp4":
            return [leaf, leaf + "_scale", leaf + "_amax", leaf + "_outer_scale_denominator"]
        if fmt == "mxfp8":
            return [leaf, leaf + "_scale"]
        return [leaf]

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        if mcore_weights_name in GLOBAL_PARAM_NAMES:
            return [GLOBAL_PARAM_NAMES[mcore_weights_name]]
        if (expert := _EXPERT_PARAM.match(mcore_weights_name)) is not None:
            layer, fc, expert_id = expert.groups()
            return self._expert_hf_names(int(layer), fc, int(expert_id))
        layer, rest = _LAYER_PARAM.match(mcore_weights_name).groups()
        prefix = f"model.llm.layers.{layer}."
        if rest in LAYER_PARAM_NAMES:
            return [prefix + LAYER_PARAM_NAMES[rest]]
        if rest == "attn_query":
            return [f"model.llm.attn_res.attn_queries.{int(layer) - 1}"]
        if rest == "mlp_query":
            return [f"model.llm.attn_res.mlp_queries.{layer}"]
        if rest == "output_query":
            return ["model.llm.attn_res.output_query"]
        if rest == "self_attention.linear_q.weight":
            return [f"{prefix}attn.{part}.weight" for part in LINEAR_Q_PARTS]
        raise NotImplementedError(f"Inkling2Bridge: unhandled parameter {mcore_weights_name}")

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        if (expert := _EXPERT_PARAM.match(mcore_weights_name)) is not None:
            # load_weights passes the rank-local name; the HF tensors hold every expert
            experts_per_rank = self.hf_config.text_config.n_routed_experts // self.mpu.ep_size
            global_expert = self.mpu.ep_rank * experts_per_rank + int(expert.group(3))
            return self._expert_to_mcore(global_expert, expert.group(2), hf_weights)
        if mcore_weights_name.endswith("self_attention.linear_q.weight"):
            layer = int(_LAYER_PARAM.match(mcore_weights_name).group(1))
            sizes = linear_q_split(self.hf_config.text_config.to_dict(), layer)
            assert [w.shape[0] for w in hf_weights] == sizes, (mcore_weights_name, sizes)
            return torch.cat(hf_weights, dim=0).to(self.dtype)
        (weight,) = hf_weights
        # attention-residual queries and the gate bias are fp32 parameters
        return weight if weight.dtype == torch.float32 else weight.to(self.dtype)

    def _expert_to_mcore(self, expert: int, fc: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        if len(hf_weights) == 4:
            weight, scale, amax, denominator = hf_weights
            weight = dequantize_nvfp4(weight[expert], scale[expert], amax[expert], denominator)
        elif len(hf_weights) == 2:
            weight, scale = hf_weights
            weight = dequantize_mxfp8(weight[expert], scale[expert])
        else:
            weight = hf_weights[0][expert].to(self.dtype)
        if fc == "2":
            return weight.contiguous()
        # HF interleaves gate and up rows; Megatron's linear_fc1 is [gate; up].
        return torch.cat((weight[0::2], weight[1::2]), dim=0).contiguous()

    def _build_config(self):
        text = self.hf_config.text_config
        return self._build_base_config(
            text_config_key="text_config",
            use_cpu_initialization=False,
            num_moe_experts=text.n_routed_experts,
            moe_router_topk=text.num_experts_per_tok,
            moe_ffn_hidden_size=text.intermediate_size,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
        )
