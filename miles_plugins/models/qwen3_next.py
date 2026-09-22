import copy

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig

from miles_plugins.models.layers.delta_rule_attention import DeltaRuleAttention, Projections
from miles_plugins.models.qwen3_5 import Attention as _Qwen3_5Attention


class Qwen3NextGatedDeltaNet(DeltaRuleAttention):
    """Qwen3-Next GDN: HF fuses ``in_proj_qkvz`` and ``in_proj_ba``, both group-major per key head, so the
    projections stay fused and are split here."""

    def _build_projections(self):
        hidden, local = self.config.hidden_size, self.local
        self.in_proj_qkvz = self.sharded_linear(
            "in_proj_qkvz", hidden, local.num_k_heads * (local.group_qkv_dim + local.group_value_dim)
        )
        self.in_proj_ba = self.sharded_linear("in_proj_ba", hidden, 2 * local.num_v_heads)

    def project(self, x):
        batch, seq_len, _ = x.shape
        local = self.local
        qkv, z = (
            self.in_proj_qkvz(x)
            .view(batch, seq_len, local.num_k_heads, -1)
            .split([local.group_qkv_dim, local.group_value_dim], dim=-1)
        )
        b, a = (
            self.in_proj_ba(x)
            .view(batch, seq_len, local.num_k_heads, -1)
            .split([local.v_per_k, local.v_per_k], dim=-1)
        )
        return Projections(
            qkv.reshape(batch, seq_len, -1),
            z.reshape(batch, seq_len, -1),
            b.reshape(batch, seq_len, -1),
            a.reshape(batch, seq_len, -1),
        )


class Attention(_Qwen3_5Attention):
    core_cls = Qwen3NextGatedDeltaNet


def get_qwen3_next_spec(args, config, vp_stage):
    # always use the moe path
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
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    # Compute layer_types if the config class doesn't expose it
    if not hasattr(hf_config, "layer_types"):
        interval = getattr(hf_config, "full_attention_interval", 4)
        n = hf_config.num_hidden_layers
        hf_config.layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)]

    for layer_id in range(num_layers_to_build):
        if hf_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
