import copy

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.transformer_block import get_num_layers_to_build

from .minimax_attention import MinimaxSelfAttention

def get_minimax_m2_spec(args, config, vp_stage):
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


    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    for layer_id in range(num_layers_to_build):
        layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
        attn_spec = layer_specs.submodules.self_attention  # 原来的 ModuleSpec
        attn_spec.module = MinimaxSelfAttention            # 只换类
        
        transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec

