"""TOP block for the Qwen3.5/3.6 GDN hybrids at every size, dense and MoE.

Every layer's ``self_attention`` is ``TopQwen35Attention``: serving's GDN (its chunk prefill per sample) on
linear-attention layers, the shared FA4 program on full-attention layers, both under HF parameter names.
The MLP is Megatron's: the stock SwiGLU MLP (dense, as Qwen3 dense TOP binds it) or the Qwen DeepEP/DeepGEMM experts
with a gated shared expert.
"""

import copy
import functools

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build

from miles.true_on_policy.gdn import qwen_gdn_model
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.top.binding import (
    NORM_ROLES,
    ROW_LINEAR_ROLES,
    TopRowParallelLinear,
    bind_dense_activation,
    bind_residual_layer,
    bind_roles,
    enable_batch_invariant,
    hf_model_type,
)


def gdn_text_config(args, config):
    """The checkpoint's text config, checked against the trainer's layer count, with ``layer_types`` filled in."""
    hf = load_hf_config(args.hf_checkpoint)
    hf = getattr(
        hf, "text_config", hf
    )  # config-access-exempt: HF configs may wrap the text model's config in text_config
    if hf.num_hidden_layers != config.num_layers:
        raise ValueError(f"[top] GDN spec has {config.num_layers} layers; the checkpoint has {hf.num_hidden_layers}")
    if not getattr(
        hf, "layer_types", None
    ):  # config-access-exempt: HF model families differ in optional layer_types metadata
        interval = hf.full_attention_interval
        hf.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(hf.num_hidden_layers)
        ]
    return hf


def _bind_gated_shared_expert(spec):
    from miles_plugins.top.moe_ep import TopGatedSharedExpert

    moe = spec.submodules.mlp.keywords["submodules"]
    shared = moe.shared_experts
    if isinstance(shared, ModuleSpec):
        shared.module = TopGatedSharedExpert
    elif isinstance(shared, functools.partial):
        moe.shared_experts = functools.partial(TopGatedSharedExpert, **shared.keywords)
    else:
        raise NotImplementedError("[top] Qwen GDN MoE expects a gated shared expert")


def gdn_block(args, config, vp_stage, stamp, *, contract):
    from miles_plugins.top.gdn_attention import TopQwen35Attention
    from miles_plugins.top.gdn_norm import TopGemmaRMSNorm

    model = qwen_gdn_model(hf_model_type(args))
    if getattr(
        config, "mtp_num_layers", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional mtp_num_layers
        raise NotImplementedError("[top] GDN TOP does not train MTP")
    hf = gdn_text_config(args, config)
    from miles_plugins.top.decode_split import bind_split

    bind_split(hf.model_type, stamp)
    if model.moe:
        # SequentialMLP owns the per-expert parameters; TOP supplies the grouped DeepGEMM arithmetic.
        config.moe_grouped_gemm = False
        if config.moe_permute_fusion:
            config.moe_permute_fusion = False
            stamp.append(("moe_permute_fusion", "True", "False (slot metadata requires unfused permutation)"))
        from miles_plugins.top.moe_ep import bind_moe_layer, validate_moe_config

        validate_moe_config(config, hf.model_type)
    elif args.backend.num_experts:
        raise ValueError("[top] the dense Qwen GDN checkpoint has no experts")
    template = get_gpt_layer_local_spec(
        num_experts=args.backend.num_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=args.backend.qk_layernorm,
        multi_latent_attention=False,
        normalization=args.backend.normalization,
    )
    if model.moe:
        bind_moe_layer(template, stamp)
        _bind_gated_shared_expert(template)
    else:
        bind_roles(template, ROW_LINEAR_ROLES, TopRowParallelLinear._build(), stamp, as_spec_param=True)
        bind_dense_activation(config, stamp)
    bind_residual_layer(template, stamp)
    bind_roles(template, NORM_ROLES, TopGemmaRMSNorm, stamp, as_spec_param=True)
    template.submodules.self_attention = ModuleSpec(module=TopQwen35Attention, params={"hf_config": hf})
    # GDN and full-attention layers own different parameters, so the distributed checkpoint keys them per layer
    # (Megatron's non-homogeneous layout) instead of stacking every layer along one axis.
    config.hetereogenous_dist_checkpoint = True
    stamp.append(("hetereogenous_dist_checkpoint", "False", "True (GDN and attention layers differ)"))
    template.submodules.sharded_state_dict_keys_map.pop("input_layernorm.", None)
    enable_batch_invariant()
    stamp.append(("hybrid_attention", "delegated", "serving's chunk prefill GDN / shared FA4 full attention"))
    stamp.append(("final_layernorm", "LayerNormImpl", TopGemmaRMSNorm.__name__))
    return TransformerBlockSubmodules(
        layer_specs=[copy.deepcopy(template) for _ in range(get_num_layers_to_build(config, vp_stage=vp_stage))],
        layer_norm=ModuleSpec(module=TopGemmaRMSNorm, params={"role": "final_layernorm"}),
    )
