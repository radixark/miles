"""Build the Megatron Step5 decoder spec from the checkpoint's per-layer layout.

Layers use gated GQA with sliding-window or sparse full attention, followed by a
clamped-SwiGLU dense or MoE block. `get_step5_spec` validates the model flags first.
"""

import argparse
from functools import partial

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TERowParallelLinear,
)
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from miles.kernels.activation.swiglu_fp32 import swiglu_fp32
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.step5.checkpoint_config import check_matches_checkpoint
from miles_plugins.models.step5.layers import (
    SWIGLU_LIMIT,
    Step5Attention,
    Step5FinalRMSNorm,
    Step5RMSNorm,
    is_full_attention_layer,
)
from miles_plugins.models.step5.sparse_attention import Step5SparseAttention, Step5SparseCoreAttention


def configure_step5(config: TransformerConfig) -> None:
    """Set what the Megatron CLI cannot: the activation callable and the fusions it rules out.

    ``min(silu(gate), 7) * clamp(up, -7, 7)`` in fp32, rounded once, for the dense MLPs, the routed experts and
    the shared expert alike: the kernel sglang's SiluClampAndMul computes, not Megatron's
    ``activation_func_clamp_value``, which clamps SiLU's input.
    """
    config.gated_activation_func = partial(swiglu_fp32, limit=SWIGLU_LIMIT)
    config.bias_activation_fusion = False
    config.use_te_activation_func = False


def build_step5_spec(
    config: TransformerConfig,
    vp_stage: int | None = None,
    *,
    reuse_indexer_on_recompute: bool = False,
) -> TransformerBlockSubmodules:
    dense_mlp = partial(
        MLP.as_mlp_submodule,
        submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
    )
    moe = get_moe_module_spec_for_backend(
        backend=TESpecProvider(), num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    sliding_window = _attention_spec(Step5Attention, core_attention=TEDotProductAttention)
    full = _attention_spec(Step5SparseAttention, core_attention=Step5SparseCoreAttention)
    full.params["reuse_indexer_on_recompute"] = reuse_indexer_on_recompute
    layer_specs = [
        _layer_spec(
            self_attention=full if is_full_attention_layer(config, layer_number) else sliding_window,
            mlp=moe if is_moe else dense_mlp,
        )
        for layer_number, is_moe in enumerate(config.moe_layer_freq, start=1)
    ]
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    num_local_layers = get_num_layers_to_build(config, vp_stage=vp_stage)
    return TransformerBlockSubmodules(
        layer_specs=layer_specs[offset : offset + num_local_layers], layer_norm=Step5FinalRMSNorm
    )


def get_step5_spec(args: argparse.Namespace, config: TransformerConfig, vp_stage: int | None):
    check_matches_checkpoint(config, load_hf_config(args.hf_checkpoint).text_config)
    configure_step5(config)
    return build_step5_spec(
        config,
        vp_stage=vp_stage,
        reuse_indexer_on_recompute=getattr(args, "reuse_indexer_on_recompute", False),
    )


def _attention_spec(module, *, core_attention) -> ModuleSpec:
    # Norms stay separate from the linears: TE's fused norm-linear keeps a bf16 norm weight.
    return ModuleSpec(
        module=module,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=TEColumnParallelLinear,
            core_attention=core_attention,
            linear_proj=TERowParallelLinear,
            q_layernorm=Step5RMSNorm,
            k_layernorm=Step5RMSNorm,
        ),
    )


def _layer_spec(*, self_attention: ModuleSpec, mlp) -> ModuleSpec:
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Step5RMSNorm,
            self_attention=self_attention,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=Step5RMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
