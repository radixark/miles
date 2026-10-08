import copy

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import gdn_heads
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.linear_attn import GatedDeltaNet, LinearAttentionLayer


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


class _TEZeroCenteredRMSNorm(te.RMSNorm):
    get_extra_state = nn.Module.get_extra_state
    set_extra_state = nn.Module.set_extra_state


def gdn_input_layernorm(kind: str, hidden_size: int, eps: float, params_dtype: torch.dtype) -> nn.Module:
    """``weight`` holds w of the ``(1 + w)`` scale under either kernel, so checkpoints and conversion see the same
    parameter."""
    if kind == "hf":
        return Qwen3NextRMSNorm(hidden_size, eps=eps)
    return _TEZeroCenteredRMSNorm(hidden_size, eps=eps, zero_centered_gamma=True, params_dtype=params_dtype)


class Qwen3_5GatedDeltaNet(GatedDeltaNet):
    """Qwen3.5 / 3.6 / 3.8 GDN: ``in_proj_qkv`` (group-major rows), ``in_proj_z``, ``in_proj_b``, ``in_proj_a``."""

    def _build_projections(self):
        hidden, local = self.config.hidden_size, self.local
        self.in_proj_qkv = self.sharded_linear("in_proj_qkv", hidden, local.num_k_heads * local.group_qkv_dim)
        self.in_proj_z = self.sharded_linear("in_proj_z", hidden, local.value_dim)
        self.in_proj_b = self.sharded_linear("in_proj_b", hidden, local.num_v_heads)
        self.in_proj_a = self.sharded_linear("in_proj_a", hidden, local.num_v_heads)

    def in_proj_weight(self):
        return torch.cat(
            [self.in_proj_qkv.weight, self.in_proj_z.weight, self.in_proj_b.weight, self.in_proj_a.weight]
        )


class Attention(LinearAttentionLayer):
    """GDN ``self_attention``; ``core_cls`` picks the family's HF projection layout."""

    core_cls: type[GatedDeltaNet] = Qwen3_5GatedDeltaNet

    def __init__(self, args, config, layer_number: int, cp_comm_type=None, pg_collection=None, name=None):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        text_config = _get_text_config(load_hf_config(args.hf_checkpoint))
        linear_attn = self.core_cls(
            config,
            heads=gdn_heads(text_config),
            conv_kernel_size=text_config.linear_conv_kernel_dim,
            norm_eps=text_config.rms_norm_eps,
            tp_group=pg_collection.tp,
            backend=args.linear_attention_backend,
            norm_activation=text_config.hidden_act,
        )
        input_layernorm = gdn_input_layernorm(
            args.linear_attention_input_norm, text_config.hidden_size, text_config.rms_norm_eps, config.params_dtype
        )
        super().__init__(config, linear_attn, input_layernorm, pg_collection, allgather_cp=args.allgather_cp)


def get_qwen3_5_spec(args, config, vp_stage):
    # always use the moe path for MoE models
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Define the decoder block spec
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    text_config = _get_text_config(hf_config)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
