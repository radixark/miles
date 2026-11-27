"""--spec entry point: admit the trainer layout, pin serving's program, and build the family's TOP block.

Each family builder binds its roles through ``binding`` and enables batch invariance after any extra overloads it
configures. Builders are named by import path in ``registry.BUILDERS``, so this module imports none of them.
"""

from __future__ import annotations

import functools
import importlib

from miles_plugins.top.binding import hf_model_type, pin_sglang, report_bindings
from miles_plugins.top.fp8_config import checkpoint_is_block_fp8
from miles_plugins.top.install import refuse_unpatched_backward
from miles_plugins.top.registry import BUILDERS, TrainerLayout, admit_trainer


def _resolve(path):
    module, name = path.split(":")
    return getattr(
        importlib.import_module(module), name
    )  # config-access-exempt: attribute selected at runtime from a table of names


def get_top_spec(args, config, vp_stage):
    """--spec entry point."""
    if not args.true_on_policy_mode:
        raise ValueError("[top] get_top_spec needs --true-on-policy-mode, which also gives the engines the contract")
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import is_batch_invariant_mode_enabled

    fp8 = checkpoint_is_block_fp8(args.hf_checkpoint)
    family = admit_trainer(
        hf_model_type(args),
        TrainerLayout.from_config(config, allgather_cp=args.allgather_cp),
        fp8=fp8,
        vision=args.multimodal_keys is not None,
    )
    refuse_unpatched_backward(config)
    build = BUILDERS[family.name]
    contract = pin_sglang(args, _resolve(build.server_kwargs)(args) if build.server_kwargs else {})
    if "packed_residual" in family.ops_for(fp8=fp8):
        from miles_plugins.top.residual_norm import validate_residual_config

        validate_residual_config(config, use_te=args.backend.transformer_impl == "transformer_engine")
    stamp: list[tuple[str, str, str]] = []
    block = _resolve(build.build)(args, config, vp_stage, stamp, contract=contract)
    report_bindings(args, contract, is_batch_invariant_mode_enabled(), stamp)
    return block


def qwen3_block(args, config, vp_stage, stamp, *, contract):
    """Qwen3 dense or Qwen3 MoE on Megatron's local layer spec."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build

    from miles_plugins.top.attention import TopAttention
    from miles_plugins.top.binding import (
        ATTENTION_ROLES,
        ROW_LINEAR_ROLES,
        TopRMSNorm,
        TopRowParallelLinear,
        bind_dense_activation,
        bind_residual_layer,
        bind_roles,
        enable_batch_invariant,
    )
    from miles_plugins.top.rope import enforce_fused_rope

    if checkpoint_is_block_fp8(args.hf_checkpoint):
        raise NotImplementedError("[top] Qwen3 TOP training requires a BF16 checkpoint")
    is_moe = hf_model_type(args) == "qwen3_moe"
    if is_moe:
        from miles_plugins.top.moe_ep import validate_moe_config

        validate_moe_config(config, "qwen3_moe")
        if args.backend.moe_grouped_gemm:
            raise NotImplementedError("[top] MoE uses SequentialMLP parameter ownership")
    stamp.append(("rope", "operation binding", enforce_fused_rope(args, config)))
    if not is_moe:
        bind_dense_activation(config, stamp)

    spec = get_gpt_layer_local_spec(
        num_experts=args.backend.num_experts,
        moe_grouped_gemm=args.backend.moe_grouped_gemm,
        qk_layernorm=args.backend.qk_layernorm,
        multi_latent_attention=args.backend.multi_latent_attention,
        normalization=args.backend.normalization,
        use_kitchen=config.use_kitchen,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )
    bind_residual_layer(spec, stamp)
    bind_roles(spec, ROW_LINEAR_ROLES, TopRowParallelLinear._build(), stamp, as_spec_param=True)
    bind_roles(spec, ATTENTION_ROLES, TopAttention, stamp)
    if is_moe:
        from megatron.core.transformer.moe.moe_layer import MoELayer

        from miles_plugins.top.grouped_experts import build_top_grouped_mlp
        from miles_plugins.top.moe_ep import TopEPMoELayer
        from miles_plugins.top.router import TopTopKRouter

        mlp = spec.submodules.mlp
        if not isinstance(mlp, functools.partial) or mlp.func is not MoELayer:
            raise NotImplementedError("[top] expected the stock MoELayer builder")
        moe = mlp.keywords["submodules"]
        moe.router = TopTopKRouter
        moe.experts = functools.partial(
            build_top_grouped_mlp(), **getattr(moe.experts, "keywords", {})
        )  # config-access-exempt: the experts builder is a functools.partial only on some layer specs
        spec.submodules.mlp = functools.partial(TopEPMoELayer, **mlp.keywords)
        # MoE keeps pre-MLP norm standalone; the dense FC1 checkpoint alias does not apply.
        spec.submodules.sharded_state_dict_keys_map.pop("pre_mlp_layernorm.", None)
        stamp.append(
            (
                "moe",
                "stock operations",
                f"shared router / DeepGEMM / slot combine (EP{config.expert_model_parallel_size})",
            )
        )
    enable_batch_invariant()

    block = TransformerBlockSubmodules(
        layer_specs=[spec] * get_num_layers_to_build(config, vp_stage=vp_stage),
        layer_norm=ModuleSpec(module=TopRMSNorm, params={"role": "final_layernorm"}),
    )
    stamp.append(("final_layernorm", "LayerNormImpl", TopRMSNorm.__name__))
    return block
