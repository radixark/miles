import copy

import torch
from megatron.core import mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig

try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention, Qwen3NextRMSNorm
except ImportError:
    Qwen3NextAttention = Qwen3NextRMSNorm = None

from .gdn_attention import GatedDeltaRuleAttentionCore, GdnLayout
from .hf_attention import HuggingfaceAttention


class Qwen3NextGatedDeltaNet(GatedDeltaRuleAttentionCore):
    """Qwen3-Next GatedDeltaNet (fused ``in_proj_qkvz`` / ``in_proj_ba``) on the unified head-sharded
    core.  With ``mp_config`` (the Megatron ``TransformerConfig``) the projections are TP-sharded by
    key-head group; without it the module is a plain replicated layer (unit tests)."""

    def __init__(self, config, layer_idx: int, args=None, *, mp_config=None, tp_group=None):
        super().__init__(
            GdnLayout.from_hf_config(config, hf_layout="qwen3_next"),
            layer_idx=layer_idx,
            gdn_backend=getattr(args, "linear_attention_backend", "fla"),
            mp_config=mp_config,
            tp_group=tp_group,
            params_dtype=config.dtype if getattr(config, "dtype", None) is not None else torch.get_default_dtype(),
            a_log_fp32=False,
        )


class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
        name: str | None = None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
            name=name,
        )
        if Qwen3NextAttention is None:
            raise ImportError("Please install transformers>=4.35.0 to use Qwen3NextAttention.")

        self.linear_attn = Qwen3NextGatedDeltaNet(
            self.hf_config, self.hf_layer_idx, args=args, mp_config=config, tp_group=mpu.get_tensor_model_parallel_group()
        )
        self.tp_sharded_compute = self.linear_attn.tp_sharded
        self.input_layernorm = Qwen3NextRMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        return hidden_states


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
