"""DeepSeek-V4.1 TOP block spec: Megatron's mHC layer flow with roles bound to SGLang's V4.1 program.

Every layer is MoE (score-routed, no hash layers). Each hyper-connection collapses the streams with the previous
one's ``pre``, carried in BLOCK_STATE through the forward; the head collapses with the last one. Engram layers add
their n-gram lookup to the streams before the layer. Layers are not recomputed: the carried ``pre`` and the index and
candidate selections other layers produced live outside the checkpointed hidden state.
"""

import functools

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer

from miles_plugins.top.binding import CaptureTap
from miles_plugins.top.dsv41_runtime import BLOCK_STATE


def dsv41_server_kwargs(args):
    """SGLang server args of the serving TOP program the trainer pins to."""
    return dict(
        kv_cache_dtype="fp8_e4m3",
        moe_a2a_backend="deepep",
        moe_runner_backend="deep_gemm",
        deepep_dispatcher_output_dtype="bf16",
        disable_shared_experts_fusion=True,
    )


def validate_dsv41_config(config):
    required = {
        "enable_hyper_connections": True,
        "num_residual_streams": 4,
        "moe_router_score_function": "sqrtsoftplus",
        "moe_router_enable_expert_bias": True,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "deepep",
        "moe_grouped_gemm": False,
        "moe_n_hash_layers": 0,
    }
    for field, value in required.items():
        if (
            getattr(config, field, None) != value
        ):  # config-access-exempt: TransformerConfig field selected at runtime from a table of checked names
            raise NotImplementedError(f"[top] DeepSeek-V4.1 requires {field}={value!r}")
    if getattr(
        config, "mtp_num_layers", None
    ):  # config-access-exempt: TransformerConfig versions and model families differ in optional mtp_num_layers
        raise NotImplementedError("[top] DeepSeek-V4.1 trains no draft layers (DSpark is serving-only)")
    if config.recompute_granularity is not None:
        raise NotImplementedError(
            "[top] DeepSeek-V4.1 layers are not recomputed: the previous hyper-connection's pre and the index and "
            "candidate selections cross layers outside the checkpointed hidden state; drop --recompute-granularity"
        )


class TopV41SharedExpert(SharedExpertMLP):
    """The serving shared expert: 32x32 FP8 gate_up, clamped SwiGLU to per-32 FP8, 32x32 FP8 down (full K)."""

    def forward(self, hidden_states):
        from miles_plugins.top.dsv4_moe import SWIGLU_LIMIT
        from miles_plugins.top.fp8_b32 import trainable_clamped_swiglu_mlp

        shape = hidden_states.shape
        out = trainable_clamped_swiglu_mlp(
            hidden_states.reshape(-1, shape[-1]), self.linear_fc1, self.linear_fc2, SWIGLU_LIMIT
        )
        return out.reshape(shape)


class TopV41HeadCollapse(MegatronModule):
    """The block's mHC exit: the last streams collapsed with the last hyper-connection's pre (no learned head)."""

    def __init__(self, config, **_):
        super().__init__(config)

    def forward(self, hidden_states):
        from miles_plugins.top.dsv41_mhc import top_collapse

        return top_collapse(hidden_states, BLOCK_STATE.pre, self.config.num_residual_streams)


@functools.cache
def _engram_hasher(device_index):
    """SGLang's EngramHasher for the pinned config (its tokenizer-built token map and hash tables), on this GPU."""
    from sglang.srt.layers.engram import EngramHasher, EngramLayout
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    from miles_plugins.top.dsv4_runtime import dsv4_hf_config

    cfg = dsv4_hf_config()
    tokenizer = get_tokenizer(get_server_args().model_path, trust_remote_code=True, tokenizer_backend="huggingface")
    hasher = EngramHasher(
        EngramLayout.from_config(cfg), tokenizer, cfg.engram_pad_token_id, cfg.engram_compressed_vocab_size
    )
    return hasher.to(torch.device("cuda", device_index))


class TopV41Layer(HyperConnectionTransformerLayer):
    """Binds each hyper-connection to its norm slot and the block state; engram layers add their lookup first."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        from miles_plugins.top.dsv4_runtime import dsv4_hf_config
        from miles_plugins.top.dsv41_mhc import TopV41HyperConnection, TopV41MhcNorm

        for hc, norm in (
            (self.self_attention_hyper_connection, self.input_layernorm),
            (self.mlp_hyper_connection, self.pre_mlp_layernorm),
        ):
            if not isinstance(hc, TopV41HyperConnection) or not isinstance(norm, TopV41MhcNorm):
                raise NotImplementedError("[top] DeepSeek-V4.1 layers require TOP V4.1 mHC and norm slots")
            hc.norm, hc.runtime = norm, BLOCK_STATE
        self.engram = None
        layer_id = self.layer_number - 1
        if layer_id in dsv4_hf_config().engram_layer_ids:
            from sglang.srt.true_on_policy.dsv41 import serves_images

            from miles_plugins.top.dsv41_engram import TopV41Engram

            cfg = dsv4_hf_config()
            self.engram = TopV41Engram(
                config,
                layer_id,
                None,
                hasher=_engram_hasher(torch.cuda.current_device()),
                image_token_id=cfg.image_token_id if serves_images(cfg) else None,
            )
        # Capture points of the per-layer VJP (top_vjp): the hyper-connection pre carried in from the previous layer
        # and out to the next, outside the layer's hidden states.
        self.top_pre_in = CaptureTap()
        self.top_pre_out = CaptureTap()

    def forward(self, hidden_states, *args, **kwargs):
        if BLOCK_STATE.pre is not None:
            BLOCK_STATE.pre = self.top_pre_in(BLOCK_STATE.pre)
        if self.engram is not None:
            hidden_states = self.engram(hidden_states, BLOCK_STATE.input_ids, BLOCK_STATE.lengths)
        out = super().forward(hidden_states, *args, **kwargs)
        BLOCK_STATE.pre = self.top_pre_out(BLOCK_STATE.pre)
        return out


def dsv41_block(args, config, vp_stage, stamp, *, contract):
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules, get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

    from miles.utils.hf_utils.config import load_hf_config
    from miles_plugins.models.deepseek_v4_1.deepseek_v4_1 import apply_v41_config
    from miles_plugins.top.binding import enable_batch_invariant
    from miles_plugins.top.dsv4_mhc import TopFinalRMSNorm
    from miles_plugins.top.dsv4_spec import _bind_sglang_tp_group
    from miles_plugins.top.dsv41_attention import TopDSV41Attention
    from miles_plugins.top.dsv41_mhc import TopV41HyperConnection, TopV41MhcNorm
    from miles_plugins.top.dsv41_moe import TopDSV41MoELayer, TopDSV41Router
    from miles_plugins.top.grouped_experts import build_top_grouped_mlp

    validate_dsv41_config(config)
    _bind_sglang_tp_group()
    # DeepSeekV41Engram (the table lookup TopV41Engram keeps) reads its layout and token map from these fields.
    hf = load_hf_config(args.hf_checkpoint)
    apply_v41_config(
        config, getattr(hf, "text_config", hf)
    )  # config-access-exempt: HF configs may wrap the text model's config in text_config
    config.v41_hf_checkpoint = args.hf_checkpoint
    from miles_plugins.top.dsv41_export import register as register_export

    register_export()
    moe = get_moe_module_spec(use_te=False, num_experts=config.num_moe_experts, moe_grouped_gemm=False)
    sub = moe.keywords["submodules"]
    sub.router = TopDSV41Router
    sub.experts = functools.partial(
        build_top_grouped_mlp(), **getattr(sub.experts, "keywords", {})
    )  # config-access-exempt: the experts builder is a functools.partial only on some layer specs
    sub.shared_experts = functools.partial(TopV41SharedExpert, **sub.shared_experts.keywords)
    layer = ModuleSpec(
        module=TopV41Layer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TopV41MhcNorm,
            self_attention=ModuleSpec(module=TopDSV41Attention),
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=TopV41HyperConnection,
            pre_mlp_layernorm=TopV41MhcNorm,
            mlp=functools.partial(TopDSV41MoELayer, **moe.keywords),
            mlp_bda=get_bias_dropout_add,
            mlp_hyper_connection=TopV41HyperConnection,
        ),
    )
    stamp.extend(
        [
            ("mhc", "HyperConnectionModule", "SGLang V4.1 mHC (prev-sublayer pre): stats, combine+norm, split-H post"),
            ("engram", "DeepSeekV41Engram", "SGLang hash kernels, exact FP8 rows, 32x32 FP8 wkv, fused gate"),
            ("self_attention", "DeepSeekV41Attention", "SGLang V4.1 attention program (private pool/backend)"),
            ("projections", "TELinear", "SGLang fp8_b32_linear (32x32 UE8M0, DeepGEMM BLOCK32)"),
            ("moe", "MoELayer", "SGLang log1p sqrtsoftplus gate, cast FP8 experts, DeepEP slot combine"),
            ("shared_expert", "SharedExpertMLP", "SGLang clamped_swiglu_b32"),
            ("hc_head", "learned_output_contract", "SGLang V4.1 head collapse (last pre)"),
        ]
    )
    block = TransformerBlockSubmodules(
        layer_specs=[layer] * get_num_layers_to_build(config, vp_stage=vp_stage),
        layer_norm=TopFinalRMSNorm,
        hc_head_contraction=TopV41HeadCollapse,
    )
    enable_batch_invariant()
    return block


def reset_block_state(module, args, kwargs):
    """GPTModel forward pre-hook: each forward starts with the micro-batch's packed ids and lengths, and no pre on
    the first pipeline stage."""
    input_ids = kwargs.get("input_ids", args[0] if args else None)
    packed = kwargs.get("packed_seq_params")
    if input_ids is None or packed is None:
        raise NotImplementedError("[top] DeepSeek-V4.1 needs the packed input ids and THD packed_seq_params")
    cu = packed.cu_seqlens_q.tolist()
    BLOCK_STATE.reset(input_ids.reshape(-1), [b - a for a, b in zip(cu[:-1], cu[1:], strict=True)])
    # A later pipeline stage starts from the pre and latents the previous stage sent (dsv41_pipeline).
    from miles_plugins.top.dsv41_pipeline import restore_block_state

    restore_block_state()
