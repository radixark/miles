"""The Cake packed causal GQA kernels as the core attention of the Qwen3.5 / Qwen3-Next full-attention layers
(``--full-attention-backend cake``).

:class:`CakeGQACoreAttention` subclasses Megatron's ``TEDotProductAttention``: it keeps its constructor and its
state-dict keys, so checkpoints load under either backend, and runs the packed (``thd``) forward and backward of
``miles_ext_kernels.gqa`` from the miles-ext-kernels submodule (head dim 256, SM100a / SM103a). The projections, the
q/k norms, RoPE and the attention output gate stay in Megatron's ``SelfAttention``.
:func:`apply_cake_full_attention` swaps it into a full-attention layer spec; the MTP layer, which Megatron builds
from the decoder's last layer spec (a full-attention layer in these models), inherits it.
:func:`validate_full_attention_backend` rejects at startup the settings the kernels do not run.

The backend is opt-in: the kernels sum dQ with FP32 atomic adds in a run-dependent order, so the backward is not
bit-reproducible, and ``--deterministic-mode`` rejects it.
"""

from __future__ import annotations

import copy

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.enums import AttnMaskType

HEAD_DIM = 256
# The --spec modules whose full-attention layers run the kernels (their spec functions call apply_cake_full_attention).
SPEC_MODULES = ("miles_plugins.models.qwen3_5", "miles_plugins.models.qwen3_next")


def _kernels():
    try:
        from miles_ext_kernels import gqa
    except ImportError as exc:
        raise ImportError(
            "--full-attention-backend cake runs miles_ext_kernels.gqa from the miles-ext-kernels submodule: pass "
            "--use-miles-ext-kernels and check the submodule out with "
            "'git submodule update --init 3rdparty/miles-ext-kernels'."
        ) from exc
    return gqa


def _attention_errors(
    *,
    kv_channels: int,
    num_attention_heads: int,
    num_query_groups: int,
    tensor_model_parallel_size: int,
    context_parallel_size: int,
    attention_dropout: float,
    window_size,
    deterministic_mode: bool,
) -> list[str]:
    """What the kernels cannot run among the attention settings that Megatron's parsed arguments and its
    ``TransformerConfig`` both carry."""
    errors = []
    if kv_channels != HEAD_DIM:
        errors.append(f"--kv-channels {kv_channels} (the kernels take head dim {HEAD_DIM})")
    # a rank holds num_attention_heads / TP query heads and, replicated below TP, max(num_query_groups, TP) / TP
    # key/value heads
    tp = tensor_model_parallel_size
    local_heads, local_groups = num_attention_heads // tp, max(num_query_groups, tp) // tp
    if local_heads % local_groups or (local_heads // local_groups) % 2:
        errors.append(
            f"{local_heads} query heads over {local_groups} key/value heads per tensor-parallel rank (the kernels "
            "need an even number of query heads per key/value head)"
        )
    if context_parallel_size != 1:
        errors.append(f"--context-parallel-size {context_parallel_size} (the kernels have no context-parallel path)")
    if attention_dropout:
        errors.append(f"--attention-dropout {attention_dropout} (the kernels have no dropout)")
    if window_size is not None:
        errors.append(f"--window-size {window_size} (the kernels are full causal)")
    if deterministic_mode:
        errors.append("--deterministic-mode (the kernels sum dQ with FP32 atomic adds in a run-dependent order)")
    return errors


def validate_full_attention_backend(args) -> None:
    """Reject at startup the settings ``--full-attention-backend cake`` cannot run.

    ``args`` is the parsed flat namespace after Megatron's own validation, which resolves ``kv_channels``.
    """
    if args.full_attention_backend != "cake":
        return
    if args.train_backend != "megatron":
        raise ValueError(f"--full-attention-backend cake needs --train-backend megatron, not {args.train_backend}")
    errors = _attention_errors(
        kv_channels=args.kv_channels,
        num_attention_heads=args.num_attention_heads,
        # without --group-query-attention every query head has its own key/value head
        num_query_groups=args.num_query_groups if args.group_query_attention else args.num_attention_heads,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        attention_dropout=args.attention_dropout,
        window_size=args.window_size,
        deterministic_mode=args.deterministic_mode,
    )
    spec = args.spec[0] if args.spec else None
    if spec not in SPEC_MODULES:
        errors.append(f"--spec {spec} (only the full-attention layers of {', '.join(SPEC_MODULES)} run the kernels)")
    if args.qkv_format != "thd":
        errors.append(f"--qkv-format {args.qkv_format} (the kernels take packed sequences, thd)")
    if args.megatron_to_hf_mode == "bridge":
        errors.append("--megatron-to-hf-mode bridge (the bridge builds its own layer specs)")
    if errors:
        raise ValueError("--full-attention-backend cake does not support " + "; ".join(errors))


class CakeGQACoreAttention(TEDotProductAttention):
    """Megatron's ``TEDotProductAttention`` running the Cake packed causal GQA kernels (``thd`` only).

    ``forward`` takes Megatron's packed ``query [T, H_q, 256]``, ``key`` / ``value [T, H_kv, 256]`` and returns
    ``[T, H_q * 256]``, as Transformer Engine does for ``thd``.
    """

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        num_splits: int | None = None,
        cp_comm_type: str | None = "p2p",
        pg_collection=None,
    ):
        errors = _attention_errors(
            kv_channels=config.kv_channels,
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_query_groups,
            tensor_model_parallel_size=config.tensor_model_parallel_size,
            context_parallel_size=config.context_parallel_size,
            attention_dropout=config.attention_dropout if attention_dropout is None else attention_dropout,
            window_size=config.window_size,
            deterministic_mode=config.deterministic_mode,
        )
        if attention_type != "self" or attn_mask_type not in (AttnMaskType.causal, AttnMaskType.padding_causal):
            errors.append(
                f"{attention_type} attention with mask {attn_mask_type} (the kernels are causal self-attention)"
            )
        if {k_channels, v_channels} - {None, HEAD_DIM}:
            errors.append(f"k/v channels {k_channels}/{v_channels} (the kernels take head dim {HEAD_DIM})")
        if config.softmax_type != "vanilla":
            errors.append(f"softmax type {config.softmax_type} (the kernels run the vanilla softmax)")
        if config.fp8_dot_product_attention:
            errors.append("FP8 dot-product attention (the kernels run BF16)")
        if config.qk_clip or config.log_max_attention_logit:
            errors.append("qk-clip / max attention logit logging (the kernels do not report the attention logits)")
        if errors:
            raise ValueError("CakeGQACoreAttention does not support " + "; ".join(errors))
        kernels = _kernels()
        kernels.device_arch()  # RuntimeError on a GPU the kernels are not built for
        super().__init__(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            k_channels=k_channels,
            v_channels=v_channels,
            num_splits=num_splits,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        # Transformer Engine's default, 1 / sqrt(head dim)
        self.cake_softmax_scale = HEAD_DIM**-0.5 if softmax_scale is None else softmax_scale
        self._varlen_gqa_attention = kernels.varlen_gqa_attention
        # build the libraries at model construction, once per process, rather than inside the first step
        kernels.prebuild()

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias=None,
        packed_seq_params=None,
        num_splits: int | None = None,
    ):
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("--full-attention-backend cake runs packed sequences only (--qkv-format thd)")
        if attention_bias is not None:
            raise ValueError("--full-attention-backend cake takes no attention bias")
        if packed_seq_params.cu_seqlens_q_padded is not None or packed_seq_params.cu_seqlens_kv_padded is not None:
            raise ValueError("--full-attention-backend cake takes no padded sequence boundaries (cu_seqlens_*_padded)")
        out = self._varlen_gqa_attention(
            query,
            key,
            value,
            packed_seq_params.cu_seqlens_q,
            packed_seq_params.cu_seqlens_kv,
            softmax_scale=self.cake_softmax_scale,
            max_seqlen_k=packed_seq_params.max_seqlen_kv,
        )
        return out.reshape(out.size(0), -1)


def apply_cake_full_attention(layer_spec):
    """A copy of the transformer-layer spec ``layer_spec`` whose self-attention runs :class:`CakeGQACoreAttention`
    in place of ``TEDotProductAttention``; ``layer_spec``, which Megatron may share between layers, is unchanged."""
    if layer_spec.submodules.self_attention.submodules.core_attention is not TEDotProductAttention:
        raise ValueError(
            "--full-attention-backend cake replaces TEDotProductAttention, but this layer's core attention is "
            f"{layer_spec.submodules.self_attention.submodules.core_attention}"
        )
    spec = copy.copy(layer_spec)
    spec.submodules = copy.copy(layer_spec.submodules)
    spec.submodules.self_attention = copy.copy(layer_spec.submodules.self_attention)
    spec.submodules.self_attention.submodules = copy.copy(layer_spec.submodules.self_attention.submodules)
    spec.submodules.self_attention.submodules.core_attention = CakeGQACoreAttention
    return spec
