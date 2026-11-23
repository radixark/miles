"""GLM-5.3-Flash (HF ``glm5_next``) TOP block: trainer TP1, experts over EP, no CP or PP.

Each layer is Megatron's mHC layer flow with roles bound to the programs serving runs: SGLang's TileLang mhc_pre
(with the sublayer RMSNorm fused) and mhc_post (``dsv4_mhc``, the ``mhc`` op), the KDA layer (``glm53_kda``) or the
DSA layer, the GLM router with DeepGEMM BF16 experts whose SwiGLU is clamped in BF16 (``glm_clamped_experts``), the
clamped dense and shared-expert MLP (serving's ``silu_and_mul_clamp``), the stream mean (``hc_mean_head``) and
SGLang's final RMSNorm. Parameter names are the native glm5_next model's, so mbridge loads the HF checkpoint as is.
"""

import functools

import torch
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec

from miles_plugins.top.kernels.delegated_linear import delegated_column_linear, delegated_row_linear
from miles_plugins.top.kernels.surrogate import surrogate
from miles_plugins.top.moe_ep import TopEPMoELayer

# GLM-5.3-Flash swiglu_limit; serving's validator refuses any other (true_on_policy.glm5_next).
SWIGLU_LIMIT = 10.0


def _clamped_swiglu_reference(gate_up, limit):
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(gate) * up).to(gate_up.dtype)


def clamped_swiglu(gate_up, limit=SWIGLU_LIMIT):
    """Serving's silu_and_mul_clamp (BF16 clamp, FP32 SiLU * up, one rounding) on [T, 2F]; a PyTorch reference of
    the same math as the gradient."""
    from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp

    def visible(gate_up):
        out = gate_up.new_empty(gate_up.shape[0], gate_up.shape[1] // 2)
        silu_and_mul_clamp(gate_up.contiguous(), out, float(limit))
        return out

    return surrogate(visible, lambda x: _clamped_swiglu_reference(x, limit), gate_up, name="clamped_swiglu_bf16")


def _clamped_mlp(x, fc1_weight, fc2_weight):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    y = delegated_row_linear(clamped_swiglu(delegated_column_linear(flat, fc1_weight)), fc2_weight)
    return y.reshape(*shape[:-1], y.shape[-1])


class TopClampedMLP(MLP):
    """The dense layers' MLP: serving's DeepseekV2MLP with swiglu_limit (BI GEMM, silu_and_mul_clamp, full-K row)."""

    def forward(self, hidden_states, per_token_scale=None, **kwargs):
        if per_token_scale is not None:
            raise NotImplementedError("[top] the GLM-5.3-Flash dense MLP takes no per-token scale")
        return _clamped_mlp(hidden_states, self.linear_fc1.weight, self.linear_fc2.weight), None


class TopClampedSharedExpert(SharedExpertMLP):
    """The shared expert: the same clamped dense MLP, ungated."""

    def forward(self, hidden_states):
        return _clamped_mlp(hidden_states, self.linear_fc1.weight, self.linear_fc2.weight)


class TopGlm53MoELayer(TopEPMoELayer):
    """The GLM EP MoE layer whose BF16 experts clamp SwiGLU (grouped_experts reads top_swiglu_limit)."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.experts.top_swiglu_limit = SWIGLU_LIMIT


class TopMeanHCHead(MegatronModule):
    """Block output contraction: serving's hc_contract, the BF16 mean over the streams under the batch-invariant
    mean; the gradient is the plain mean's."""

    def __init__(self, config, **_):
        super().__init__(config)
        self.n = config.num_residual_streams

    def forward(self, hidden_states):
        from sglang.srt.batch_invariant_ops.batch_invariant_ops import mean_batch_invariant

        s, b, width = hidden_states.shape

        def visible(x):
            return mean_batch_invariant(x.reshape(s * b, self.n, width // self.n), [-2])

        def reference(x):
            return x.reshape(s * b, self.n, width // self.n).float().mean(dim=-2).to(x.dtype)

        return surrogate(visible, reference, hidden_states, name="hc_mean_head").view(s, b, -1)


def _glm53_layer_cls():
    from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer

    from miles_plugins.top.dsv4_mhc import TopHyperConnection, TopMhcNorm

    class TopGlm53Layer(HyperConnectionTransformerLayer):
        """Binds each mHC module to the norm slot its fused mhc_pre applies, as dsv4_mhc.TopDSV4Layer does, but
        without registering the norm a second time under the mHC module: mbridge loads parameters by name."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for hc, norm in (
                (self.self_attention_hyper_connection, self.input_layernorm),
                (self.mlp_hyper_connection, self.pre_mlp_layernorm),
            ):
                if not isinstance(hc, TopHyperConnection) or not isinstance(norm, TopMhcNorm):
                    raise NotImplementedError("[top] GLM-5.3-Flash layers require TOP mHC and norm slots")
                object.__setattr__(hc, "norm", norm)

    return TopGlm53Layer


def glm53_server_kwargs(args):
    """SGLang server args of the serving TOP program the trainer pins to (true_on_policy.kda_chunked, glm5_next and
    the DSA op's)."""
    return dict(
        attention_backend="dsa",
        dsa_prefill_backend="tilelang",
        dsa_decode_backend="tilelang",
        page_size=256,
        kv_cache_dtype="bfloat16",
        moe_a2a_backend="deepep",
        moe_runner_backend="deep_gemm",
        deepep_dispatcher_output_dtype="bf16",
        disable_shared_experts_fusion=True,
        linear_attn_backend="triton",
        mamba_ssm_dtype="float32",
        disable_radix_cache=True,
    )


def _hf_text_config(args):
    from miles.utils.hf_utils.config import load_hf_config

    hf = load_hf_config(args.hf_checkpoint)
    return getattr(hf, "text_config", None) or hf  # config-access-exempt: HF configs may wrap text_config


def validate_glm53_config(config, hf):
    """Megatron options the TOP layers would silently drop or that differ from the checkpoint."""
    from miles_plugins.top.glm_config import refuse_ignored_router_losses

    refuse_ignored_router_losses(config)
    if config.moe_router_topk_scaling_factor != hf.routed_scaling_factor:
        raise ValueError(
            f"[top] GLM-5.3-Flash routed scaling {config.moe_router_topk_scaling_factor} differs from the "
            f"checkpoint's {hf.routed_scaling_factor} (--moe-router-topk-scaling-factor)"
        )
    if getattr(
        config, "moe_router_dtype", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional moe_router_dtype
        raise NotImplementedError(
            "[top] GLM-5.3-Flash routes at serving's precision (BF16 gate logits); drop --moe-router-dtype"
        )
    if getattr(
        config, "mtp_num_layers", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional mtp_num_layers
        raise NotImplementedError("[top] GLM-5.3-Flash TOP does not train MTP")
    if config.context_parallel_size != 1 or config.pipeline_model_parallel_size != 1:
        raise NotImplementedError("[top] GLM-5.3-Flash TOP trains at CP1 and PP1")


def glm53_block(args, config, vp_stage, stamp, *, contract):
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules, get_transformer_layer_offset

    from miles_plugins.models.glm5_next.glm5_next import _apply_glm5_next_config, _patch_reference_proj_rms
    from miles_plugins.top.binding import enable_batch_invariant
    from miles_plugins.top.dsv4_mhc import TopFinalRMSNorm, TopHyperConnection, TopMhcNorm
    from miles_plugins.top.dsv4_spec import _bind_sglang_tp_group
    from miles_plugins.top.glm53_dsa import dsa_attention_spec
    from miles_plugins.top.glm53_kda import TopGlm5NextKDAAttention
    from miles_plugins.top.glm_router import TopGLMRouter
    from miles_plugins.top.grouped_experts import build_top_grouped_mlp

    hf = _hf_text_config(args)
    if hf.model_type != "glm5_next_text":
        raise ValueError("[top] the GLM-5.3-Flash TOP spec needs a glm5_next checkpoint")
    _apply_glm5_next_config(config, hf)
    # The mHC gradient is Megatron's compute_mappings: its RMS must be serving's rsqrt(mean(x^2) + layernorm eps),
    # as the native GLM-5.3 spec patches it, not Megatron's 1 / (rms + 1e-6) (layer 0's embedding rows are small).
    _patch_reference_proj_rms()
    # SequentialMLP owns the per-expert parameters; TOP supplies the grouped DeepGEMM arithmetic.
    config.moe_grouped_gemm = False
    if config.moe_permute_fusion:
        config.moe_permute_fusion = False
        stamp.append(("moe_permute_fusion", "True", "False (slot metadata requires unfused permutation)"))
    validate_glm53_config(config, hf)
    _bind_sglang_tp_group()

    layer_cls = _glm53_layer_cls()
    moe = get_moe_module_spec(use_te=False, num_experts=config.num_moe_experts, moe_grouped_gemm=False)
    sub = moe.keywords["submodules"]
    sub.router = TopGLMRouter
    sub.experts = functools.partial(
        build_top_grouped_mlp(), **getattr(sub.experts, "keywords", {})
    )  # config-access-exempt: the experts builder is a functools.partial only on some layer specs
    sub.shared_experts = functools.partial(TopClampedSharedExpert, **sub.shared_experts.keywords)
    moe_mlp = functools.partial(TopGlm53MoELayer, **moe.keywords)
    # as_mlp_submodule maps the layer's pg_collection to MLP's tp_group (the layer rewraps only a plain MLP spec).
    dense_mlp = functools.partial(
        TopClampedMLP.as_mlp_submodule,
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
    )
    kda = ModuleSpec(module=TopGlm5NextKDAAttention, params={"hf_config": hf})
    dsa = dsa_attention_spec(args, config, hf)
    dsa_layers = set(config.glm5_next_full_attn_layers)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    layers = []
    for i in range(get_num_layers_to_build(config, vp_stage=vp_stage)):
        layer_id = offset + i
        dense = layer_id < hf.first_k_dense_replace
        layers.append(
            ModuleSpec(
                module=layer_cls,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=TopMhcNorm,
                    self_attention=dsa if layer_id in dsa_layers else kda,
                    self_attn_bda=get_bias_dropout_add,
                    self_attention_hyper_connection=TopHyperConnection,
                    pre_mlp_layernorm=TopMhcNorm,
                    mlp=dense_mlp if dense else moe_mlp,
                    mlp_bda=get_bias_dropout_add,
                    mlp_hyper_connection=TopHyperConnection,
                ),
            )
        )
    stamp.extend(
        [
            ("mhc", "HyperConnectionModule", "SGLang mhc_pre(+norm)/mhc_post; gradient RMS rsqrt(mean(x^2) + eps)"),
            ("hc_head", "output_contract", "SGLang batch-invariant stream mean"),
            ("kda", "Glm5NextKDAAttention", "SGLang pinned chunk_kda (decode follows it)"),
            ("dsa", "Glm5NextDSAAttention", "SGLang GLM-5.3-Flash DSA program"),
            (
                "moe",
                "MoELayer",
                "GLM sigmoid gate, clamped BF16 DeepGEMM experts, clamped shared expert, slot combine",
            ),
            ("dense_mlp", "MLP", "SGLang silu_and_mul_clamp"),
        ]
    )
    block = TransformerBlockSubmodules(
        layer_specs=layers,
        layer_norm=TopFinalRMSNorm,
        hc_head_contraction=TopMeanHCHead,
    )
    enable_batch_invariant()
    return block
