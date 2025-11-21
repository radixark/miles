from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.transformer.attention import SelfAttentionSubmodules


def get_qwen3_next_spec(args, config, vp_stage):
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

    for layer_id in range(num_layers_to_build):
        layer_spec = transformer_layer_spec.layer_specs[layer_id]
        self_attention_submodules = layer_spec.submodules.self_attention.submodules
        assert isinstance(self_attention_submodules, SelfAttentionSubmodules)
        for k, num_heads in [
            ("q_layernorm", TODO),
            ("k_layernorm", TODO),
        ]:
            v_old = getattr(self_attention_submodules, k)
            v_new = _create_per_layer_rms_norm(v_old, num_heads=num_heads)
            setattr(self_attention_submodules, k, v_new)

    return transformer_layer_spec


def _create_per_layer_rms_norm(inner_cls: type, num_heads: int) -> type:
    return ModuleSpec(
        module=_PerLayerRMSNorm,
        params=dict(
            inner_cls=inner_cls,
            num_heads=num_heads,
        ),
    )


class _PerLayerRMSNorm:
    def __init__(self, *args, hidden_size: int, inner_cls: type, num_heads: int, **kwargs):
        self._inner = inner_cls(*args, hidden_size=hidden_size * num_heads, **kwargs)
