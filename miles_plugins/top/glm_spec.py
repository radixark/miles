"""GLM-5 (glm_moe_dsa) TOP block: the stock GLM-5 spec with roles bound to SGLang ops."""

import functools

from megatron.core.transformer.spec_utils import ModuleSpec

from miles_plugins.top.binding import (
    ROW_LINEAR_ROLES,
    TopRMSNorm,
    TopRowParallelLinear,
    bind_dense_activation,
    bind_residual_layer,
    bind_roles,
    enable_batch_invariant,
)


def glm_server_kwargs(args):
    """SGLang server args of the serving TOP program the trainer pins to."""
    from miles_plugins.top.fp8_config import checkpoint_is_block_fp8

    return dict(
        attention_backend="dsa",
        kv_cache_dtype="fp8_e4m3" if checkpoint_is_block_fp8(args.hf_checkpoint) else "bfloat16",
        dsa_prefill_backend="flashmla_sparse",
        dsa_decode_backend="flashmla_sparse",
        dsa_topk_backend="flashinfer",
        page_size=64,
    )


def glm_block(args, config, vp_stage, stamp, *, contract):
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from miles_plugins.models.glm5.glm5 import get_glm5_spec
    from miles_plugins.top.glm_config import prepare_glm_config
    from miles_plugins.top.glm_norm import TopLayerNorm
    from miles_plugins.top.glm_router import TopGLMRouter
    from miles_plugins.top.moe_ep import bind_moe_layer

    permute_fusion_requested = config.moe_permute_fusion
    prepare_glm_config(args, config, contract=contract)
    if permute_fusion_requested:
        stamp.append(("moe_permute_fusion", "True", "False (slot metadata requires unfused permutation)"))
    stamp.append(("rope", "operation binding", "SGLang fused rotary"))
    bind_dense_activation(config, stamp)
    block = get_glm5_spec(args, config, vp_stage)
    for layer in block.layer_specs:
        mlp = layer.submodules.mlp
        if isinstance(mlp, functools.partial) and mlp.func is MoELayer:
            bind_moe_layer(layer, stamp, router_cls=TopGLMRouter)
        else:
            layer.submodules.sharded_state_dict_keys_map["pre_mlp_layernorm."] = "mlp.linear_fc1.layer_norm_"
        bind_residual_layer(layer, stamp)
        bind_roles(layer, ("kv_layernorm",), TopRMSNorm, stamp, as_spec_param=True)
        bind_roles(layer, ("k_norm",), TopLayerNorm, stamp, as_spec_param=True)
        # Match the checkpoint keys written by the ordinary fused TE projections.
        layer.submodules.sharded_state_dict_keys_map.update(
            {
                "self_attention.q_layernorm.": "self_attention.linear_q_up_proj.layer_norm_",
                "self_attention.kv_layernorm.": "self_attention.linear_kv_up_proj.layer_norm_",
            }
        )
        bind_roles(layer, ROW_LINEAR_ROLES, TopRowParallelLinear._build(), stamp, as_spec_param=True)
    # Both checkpoint formats use the same absorbed-MLA contractions.
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import configure_batch_invariant_bmm
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_batch_invariant

    configure_batch_invariant_bmm(bmm_batch_invariant)
    stamp.append(("absorbed_mla_bmm", "operation binding", "SGLang batch-invariant bmm"))
    enable_batch_invariant()
    block.layer_norm = ModuleSpec(module=TopRMSNorm, params={"role": "final_layernorm"})
    stamp.append(("final_layernorm", "stock", "packed residual / SGLang RMSNorm"))
    return block
