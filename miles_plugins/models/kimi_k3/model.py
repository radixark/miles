import copy

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec

from .layers import KimiK3Attention, KimiK3TransformerLayer
from .ops import situ_and_mul


KIMI_K3_KDA_LAYERS = tuple(
    layer_number for layer_number in range(1, 94) if layer_number % 4 != 0 and layer_number != 93
)


def configure_kimi_k3(config) -> None:
    assert config.num_layers <= 93
    assert config.num_attention_heads == 96
    assert config.moe_latent_size == 3584
    config.kimi_kda_layers = tuple(
        layer_number for layer_number in KIMI_K3_KDA_LAYERS if layer_number <= config.num_layers
    )
    config.kimi_linear_num_heads = 96
    config.kimi_linear_head_dim = 128
    config.kimi_linear_conv_kernel_size = 4
    config.kimi_kda_gate_lower_bound = -5.0
    config.kimi_attn_res_block_size = 12
    config.gated_activation_func = situ_and_mul
    config.moe_latent_use_norm = True
    config.bias_activation_fusion = False
    config.use_te_activation_func = False


def build_kimi_k3_spec(config, vp_stage=None):
    assert config.pipeline_model_parallel_size == 1, "Kimi K3 initially requires PP=1"
    assert config.virtual_pipeline_model_parallel_size is None, "Kimi K3 does not support VPP yet"

    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )
    layer_specs = []
    for layer_spec in block_spec.layer_specs:
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.module = KimiK3TransformerLayer
        layer_spec.submodules.self_attention = ModuleSpec(module=KimiK3Attention)
        layer_spec.submodules.input_layernorm = TENorm
        layer_spec.submodules.pre_mlp_layernorm = TENorm

        if not config.moe_layer_freq[len(layer_specs)]:
            layer_spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear

        layer_specs.append(layer_spec)

    block_spec.layer_specs = layer_specs
    return block_spec


def get_kimi_k3_spec(args, config, vp_stage):
    del args
    configure_kimi_k3(config)
    return build_kimi_k3_spec(config, vp_stage=vp_stage)
