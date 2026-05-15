"""Chunked hidden-states -> log-probs operator.

Computes per-token log-probabilities from transformer hidden states without
materializing the full ``[T, V]`` logits tensor.  Processing is done in
configurable chunks so that peak logits memory is ``O(chunk_size * V)``
instead of ``O(seq_len * V)``.

Parallelism support
-------------------
==== ============================================================
TP=1  Per-chunk loop with activation checkpointing —
      logits recomputed in backward, peak memory O(chunk*V).
TP>1  Per-chunk loop with activation checkpointing —
      ``ColumnParallelLinear`` + TP all-reduce re-executes.
CP    Handled by :func:`chunked_log_probs_from_hidden_states`
      (zigzag offset slicing, same logic as ``get_responses``).
SP    Supported — ``gather_from_sequence_parallel_region`` before
      per-sample loop, ``output_layer.sequence_parallel`` toggled
      via try/finally.
PP    Transparent — only the pipeline-last stage calls this.
====  ============================================================

Forward / backward consistency invariant
----------------------------------------
The actor's no-grad forward (``forward_only`` -> ``actor_log_probs``) and the
training forward (``train_one_step`` -> ``policy_loss``) BOTH route through
``chunked_log_probs``.  By construction the per-chunk projection is performed
by a single shared helper (:func:`_project_chunk_to_logits`) and a single
log-prob kernel (:func:`_log_probs_from_logits`), so given **identical
hidden-state inputs** the two paths produce **bit-identical** log-probs.

Therefore any non-zero ``train_rollout_logprob_abs_diff`` /
``LOGDIFF EXTREME`` reading does **not** originate from the chunk operator
itself; it originates from the upstream ``model(...)`` forward pass running
under different contexts (``model.eval()`` + ``no_grad`` vs ``model.train()``
+ ``activation checkpointing``) or from the SGLang inference engine
(Triton FA3, bf16 throughout) vs Megatron TE (cuDNN FA2) numerics.

Memory / dtype knobs
--------------------
* ``MILES_CHUNKED_LOGPROBS_KEEP_BF16=1`` — when flash-attn cross-entropy is
  available, keep per-chunk logits in ``weight.dtype`` (bf16/fp16) instead of
  upcasting to fp32 before the kernel.  flash CE accumulates softmax in fp32
  internally, so this is numerically equivalent for the logsumexp / gather
  outputs while halving the per-chunk logits buffer (saves ``chunk_size * V``
  bytes peak).  Kept as opt-in env so existing fp32 runs stay bit-identical.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .parallel import ParallelState

logger = logging.getLogger(__name__)

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _flash_cross_entropy

    _HAS_FLASH_CE = True
except ImportError:
    _HAS_FLASH_CE = False


def _keep_low_precision_logits(*, with_entropy: bool) -> bool:
    """Whether per-chunk logits may stay in ``weight.dtype`` instead of fp32.

    Only safe when (a) flash-attn cross-entropy is available — its Triton
    kernel handles the fp32 softmax internally — and (b) we don't need the
    same logits for entropy, whose ``F.softmax`` + ``(p * logits).sum``
    formulation is precision-sensitive in bf16 over a 130k-vocab.

    Gated on an env var so the default behavior matches the prior fp32 path
    bit-for-bit; flip the env to opt into the memory win after A/B verification.
    """
    if not _HAS_FLASH_CE:
        return False
    if with_entropy:
        return False
    return os.environ.get("MILES_CHUNKED_LOGPROBS_KEEP_BF16", "0").strip().lower() in {"1", "true", "yes"}


# =====================================================================
# Internal helpers
# =====================================================================


def _log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs.  Prefers flash-attn fused cross-entropy.

    The flash-attn Triton kernel accepts fp16/bf16/fp32 logits and computes
    the softmax accumulation in fp32 internally, so passing low-precision
    logits is numerically equivalent to upcasting beforehand while halving
    the per-chunk logits buffer.  The pure-PyTorch fallback path uses
    ``logits.logsumexp`` which is precision-sensitive; when flash CE is
    unavailable the caller is responsible for handing in fp32 logits.
    """
    if _HAS_FLASH_CE:
        batch_dim = logits.shape[:-1]
        losses, _ = _flash_cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            inplace_backward=True,
        )
        return (-losses).view(*batch_dim)
    gathered = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return gathered - logits.logsumexp(-1)


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy: ``H = logsumexp(x) - sum(softmax(x) * x)``."""
    logits = logits.float()
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - (pd * logits).sum(dim=-1)


def _project_chunk_to_logits(
    hs: torch.Tensor,
    weight: torch.Tensor,
    temperature: float,
    *,
    keep_low_precision: bool,
) -> torch.Tensor:
    """Project a single TP=1 chunk of hidden states to logits.

    Single source of truth for the per-chunk projection used by **both**
    the no-grad actor path (:func:`_log_probs_no_grad_tp1`) and the
    grad-enabled training path (:func:`_chunk_log_probs_fn`).  Sharing
    the kernel guarantees the two paths produce bit-identical logits on
    identical ``(hs, weight)`` inputs — see the module-level "Forward /
    backward consistency invariant" docstring.

    Args:
        hs:                 ``[chunk, H]`` hidden states (any float dtype).
        weight:             ``[V, H]`` LM-head weight.
        temperature:        Logit temperature; in-place divide is safe
            because ``logits`` is freshly allocated by this helper and
            has no other consumers.
        keep_low_precision: When True the returned logits keep
            ``weight.dtype`` (typically bf16); see
            :func:`_keep_low_precision_logits` for when this is safe.
            When False the logits are upcast to fp32 (legacy behavior).

    Returns:
        ``[chunk, V]`` logits in fp32 or ``weight.dtype``.
    """
    if hs.dtype != weight.dtype:
        hs = hs.to(weight.dtype)
    logits = F.linear(hs, weight)
    if not keep_low_precision and logits.dtype != torch.float32:
        logits = logits.float()
    if temperature != 1.0:
        logits.div_(temperature)
    return logits


def _project_to_fp32_logits(
    hidden_states: torch.Tensor,
    output_layer,
    output_weight: torch.Tensor | None,
    tp_size: int,
    temperature: float,
) -> torch.Tensor:
    """Project hidden states → fp32 logits with temperature scaling.

    TP=1 uses ``F.linear``; TP>1 delegates to ``output_layer``
    (``ColumnParallelLinear``).  Always returns fp32 — used by the entropy
    path which needs precise softmax over the full vocab.
    """
    if tp_size == 1:
        weight = output_weight if output_weight is not None else output_layer.weight
        return _project_chunk_to_logits(
            hidden_states,
            weight,
            temperature,
            keep_low_precision=False,
        )
    if output_weight is not None and hidden_states.dtype != output_weight.dtype:
        hidden_states = hidden_states.to(dtype=output_weight.dtype, copy=False)
    logits, _ = output_layer(hidden_states, weight=output_weight)
    if logits.dtype != torch.float32:
        logits = logits.float()
    if temperature != 1.0:
        logits.div_(temperature)
    return logits


# =====================================================================
# TP=1: per-chunk forward with activation checkpointing
# =====================================================================


def _chunk_log_probs_fn(
    hs: torch.Tensor,
    lbl: torch.Tensor,
    weight: torch.Tensor,
    temperature: float,
    keep_low_precision: bool,
) -> torch.Tensor:
    """Project one chunk of hidden states to log-probs.

    Separated so ``torch.utils.checkpoint.checkpoint`` can wrap it:
    logits are recomputed during backward instead of being retained,
    keeping peak logits memory at O(chunk_size * V) regardless of the
    total number of chunks.

    Delegates to :func:`_project_chunk_to_logits` (shared with the no-grad
    path) so the with-grad and no-grad outputs are bit-identical.
    """
    logits = _project_chunk_to_logits(
        hs,
        weight,
        temperature,
        keep_low_precision=keep_low_precision,
    )
    return _log_probs_from_logits(logits, lbl)


def _chunked_log_probs_tp1_with_grad(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int,
    temperature: float,
    *,
    keep_low_precision: bool,
) -> torch.Tensor:
    """TP=1 grad-enabled path with activation checkpointing.

    Each chunk projects ``hidden_states → logits → log_probs``.  The
    projection is wrapped in ``torch.utils.checkpoint`` so that logits
    are **not** retained for backward — they are recomputed from the
    saved (hidden_states_chunk, weight) inputs.  This keeps peak logits
    memory at ``O(chunk_size * V)`` instead of ``O(response_len * V)``.
    """
    from torch.utils.checkpoint import checkpoint as ckpt

    R = hidden_states.size(0)
    lp_parts: list[torch.Tensor] = []

    for s in range(0, R, chunk_size):
        e = min(s + chunk_size, R)
        lp_parts.append(
            ckpt(
                _chunk_log_probs_fn,
                hidden_states[s:e],
                labels[s:e],
                weight,
                temperature,
                keep_low_precision,
                use_reentrant=False,
            )
        )

    return torch.cat(lp_parts) if len(lp_parts) > 1 else lp_parts[0]


# =====================================================================
# TP=1 no-grad: single-pass log-probs (+ optional entropy)
# =====================================================================


def _log_probs_no_grad_tp1(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int,
    temperature: float,
    with_entropy: bool = False,
    *,
    keep_low_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """TP=1 inference path: log-probs and optional entropy in one chunk pass.

    Uses the same :func:`_project_chunk_to_logits` helper as the
    grad-enabled path so the no-grad and with-grad outputs are
    bit-identical on identical hidden-state inputs.  When ``with_entropy``
    is True, ``keep_low_precision`` is forced off because the entropy
    formula's ``F.softmax`` + ``(p * logits).sum`` path is precision-
    sensitive in bf16.
    """
    R = hidden_states.size(0)
    log_probs = torch.empty(R, device=hidden_states.device, dtype=torch.float32)
    entropy = torch.empty(R, device=hidden_states.device, dtype=torch.float32) if with_entropy else None

    keep_lp = keep_low_precision and entropy is None

    for s in range(0, R, chunk_size):
        e = min(s + chunk_size, R)
        logits = _project_chunk_to_logits(
            hidden_states[s:e],
            weight,
            temperature,
            keep_low_precision=keep_lp,
        )
        log_probs[s:e] = _log_probs_from_logits(logits, labels[s:e])
        if entropy is not None:
            entropy[s:e] = _entropy_from_logits(logits)
        del logits

    return log_probs, entropy


# =====================================================================
# TP>1: per-chunk forward with activation checkpointing
# =====================================================================


def _chunked_log_probs_tp_parallel(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    output_layer,
    output_weight: torch.Tensor | None,
    chunk_size: int,
    temperature: float,
    tp_group,
    true_on_policy: bool = False,
    with_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked log-probs for TP>1 with activation checkpointing.

    Grad-enabled chunks are wrapped in ``torch.utils.checkpoint`` so
    logits are recomputed in backward (TP all-reduce re-executes).
    Peak logits memory stays at ``O(chunk_size * V)``.

    No-grad chunks compute log-probs (+ optional entropy) in a single
    pass per chunk with explicit ``del logits``.
    """
    from torch.utils.checkpoint import checkpoint as ckpt

    from miles.utils.ppo_utils import calculate_log_probs_and_entropy

    R = hidden_states.size(0)
    lp_parts: list[torch.Tensor] = []
    ent_parts: list[torch.Tensor] = []

    def _project_tp(_hs: torch.Tensor) -> torch.Tensor:
        """Shared TP>1 projection: ``ColumnParallelLinear`` → fp32 logits.

        Single source of truth for the with-grad and no-grad paths so the
        two paths produce bit-identical logits on identical inputs.  The
        fp32 upcast and in-place temperature divide are kept here (TP>1
        ``compute_log_probs`` / ``compute_entropy_from_logits`` go through
        ``fused_vocab_parallel_cross_entropy`` / ``_VocabParallelEntropy``
        which expect fp32-quality input for the cross-rank reductions).
        """
        if output_weight is not None and _hs.dtype != output_weight.dtype:
            _hs = _hs.to(dtype=output_weight.dtype, copy=False)
        logits, _ = output_layer(_hs, weight=output_weight)
        if logits.dtype != torch.float32:
            logits = logits.float()
        if temperature != 1.0:
            logits.div_(temperature)
        return logits

    def _project_lp_only(
        _hs: torch.Tensor,
        _tk: torch.Tensor,
    ) -> torch.Tensor:
        _was_sp = getattr(output_layer, "sequence_parallel", False)
        if _was_sp:
            output_layer.sequence_parallel = False
        try:
            logits = _project_tp(_hs)
            lp, _ = calculate_log_probs_and_entropy(
                logits,
                _tk,
                tp_group,
                with_entropy=False,
                chunk_size=-1,
                true_on_policy=true_on_policy,
            )
        finally:
            if _was_sp:
                output_layer.sequence_parallel = True
        return lp.squeeze(-1) if lp.dim() > 1 else lp

    def _project_and_compute_no_grad(
        _hs: torch.Tensor,
        _tk: torch.Tensor,
        _want_ent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _was_sp = getattr(output_layer, "sequence_parallel", False)
        if _was_sp:
            output_layer.sequence_parallel = False
        try:
            logits = _project_tp(_hs)
            lp, ent = calculate_log_probs_and_entropy(
                logits,
                _tk,
                tp_group,
                with_entropy=_want_ent,
                chunk_size=-1,
                true_on_policy=true_on_policy,
            )
            del logits
        finally:
            if _was_sp:
                output_layer.sequence_parallel = True
        lp = lp.squeeze(-1) if lp.dim() > 1 else lp
        if ent is not None:
            ent = ent.squeeze(-1) if ent.dim() > 1 else ent
        return lp, ent

    compute_entropy_inline = with_entropy and not torch.is_grad_enabled()

    for s in range(0, R, chunk_size):
        e = min(s + chunk_size, R)
        hs_c = hidden_states[s:e]
        tk_c = labels[s:e]

        if torch.is_grad_enabled():
            lp_parts.append(ckpt(_project_lp_only, hs_c, tk_c, use_reentrant=False))
        else:
            lp, ent = _project_and_compute_no_grad(hs_c, tk_c, compute_entropy_inline)
            lp_parts.append(lp)
            if ent is not None:
                ent_parts.append(ent)

    log_probs = torch.cat(lp_parts, dim=0)
    entropy = torch.cat(ent_parts, dim=0) if ent_parts else None
    return log_probs, entropy


# =====================================================================
# Entropy (always no_grad, shared by all TP paths)
# =====================================================================


def _compute_entropy_chunked(
    hidden_states: torch.Tensor,
    output_layer,
    output_weight: torch.Tensor | None,
    chunk_size: int,
    temperature: float,
    tp_size: int,
    tp_group,
) -> torch.Tensor:
    R = hidden_states.size(0)
    if R == 0:
        return hidden_states.new_zeros(0, dtype=torch.float32)

    parts: list[torch.Tensor] = []
    for s in range(0, R, chunk_size):
        e = min(s + chunk_size, R)
        logits = _project_to_fp32_logits(
            hidden_states[s:e],
            output_layer,
            output_weight,
            tp_size,
            temperature,
        )
        if tp_size == 1:
            parts.append(_entropy_from_logits(logits))
        else:
            from miles.utils.ppo_utils import compute_entropy_from_logits

            parts.append(compute_entropy_from_logits(logits, tp_group))
        del logits

    return torch.cat(parts, dim=0)


# =====================================================================
# Public API — per-sample operator
# =====================================================================


def chunked_log_probs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_layer,
    output_weight: torch.Tensor | None,
    chunk_size: int = 2048,
    temperature: float = 1.0,
    tp_size: int = 1,
    tp_group=None,
    with_entropy: bool = False,
    true_on_policy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute per-token log-probs from hidden states in chunks.

    This is the **per-sample** operator.  It receives the already-sliced
    response hidden states for a single sample and returns log-probs
    (and optionally entropy).  CP / per-sample iteration is the caller's
    responsibility — see :func:`chunked_log_probs_from_hidden_states`.

    Args:
        hidden_states: ``[R, H]`` response hidden states.
        labels:        ``[R]`` target token ids.
        output_layer:  LM head (``ColumnParallelLinear``).
        output_weight: Shared embedding weight, or ``None``.
        chunk_size:    Tokens processed per chunk.
        temperature:   Logit temperature.
        tp_size:       Tensor-parallel world size.
        tp_group:      TP process group (required when ``tp_size > 1``).
        with_entropy:  Also compute entropy (always under ``no_grad``).
        true_on_policy: Use ``log_softmax`` path (forwarded to TP>1 only).

    Returns:
        ``(log_probs [R], entropy [R] | None)`` — both fp32.
    """
    R = hidden_states.size(0)
    if R == 0:
        empty = hidden_states.new_zeros(0, dtype=torch.float32)
        return empty, (empty if with_entropy else None)

    keep_low_precision = _keep_low_precision_logits(with_entropy=with_entropy)

    if tp_size == 1:
        weight = output_weight if output_weight is not None else output_layer.weight
        if torch.is_grad_enabled():
            log_probs = _chunked_log_probs_tp1_with_grad(
                hidden_states,
                labels,
                weight,
                chunk_size,
                temperature,
                keep_low_precision=keep_low_precision,
            )
            entropy = None
            if with_entropy:
                with torch.no_grad():
                    entropy = _compute_entropy_chunked(
                        hidden_states,
                        output_layer,
                        output_weight,
                        chunk_size,
                        temperature,
                        tp_size,
                        tp_group,
                    )
        else:
            log_probs, entropy = _log_probs_no_grad_tp1(
                hidden_states,
                labels,
                weight,
                chunk_size,
                temperature,
                with_entropy,
                keep_low_precision=keep_low_precision,
            )
    else:
        if torch.is_grad_enabled():
            log_probs, _ = _chunked_log_probs_tp_parallel(
                hidden_states,
                labels,
                output_layer,
                output_weight,
                chunk_size,
                temperature,
                tp_group,
                true_on_policy=true_on_policy,
                with_entropy=False,
            )
            entropy = None
            if with_entropy:
                with torch.no_grad():
                    entropy = _compute_entropy_chunked(
                        hidden_states,
                        output_layer,
                        output_weight,
                        chunk_size,
                        temperature,
                        tp_size,
                        tp_group,
                    )
        else:
            log_probs, entropy = _chunked_log_probs_tp_parallel(
                hidden_states,
                labels,
                output_layer,
                output_weight,
                chunk_size,
                temperature,
                tp_group,
                true_on_policy=true_on_policy,
                with_entropy=with_entropy,
            )

    return log_probs, entropy


# =====================================================================
# Public API — full replacement (handles CP / SP / per-sample loop)
# =====================================================================


def chunked_log_probs_from_hidden_states(
    hidden_states: torch.Tensor,
    *,
    args: Namespace,
    parallel_state: ParallelState,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    max_seq_lens: list[int] | None = None,
    output_layer,
    output_weight,
) -> dict[str, list[torch.Tensor]]:
    """Compute chunked log-probs from packed hidden states.

    Drop-in replacement for the former
    ``_get_log_probs_from_hidden_states_chunked``.  Handles per-sample
    iteration, CP zigzag offsets, and delegates the actual chunk
    computation to :func:`chunked_log_probs`.

    Args:
        hidden_states: Model output — ``[1, T, H]`` (thd) or ``[S, B, H]``
            (bshd).  When ``log_probs_chunk_size > 0``, the model's
            ``output_layer`` has been bypassed, so this is raw hidden
            states, not logits.
        args:             Config namespace.
        parallel_state:   ``ParallelState`` (tp / cp / dp info).
        unconcat_tokens:  Per-sample token tensors.
        total_lengths:    Per-sample total (prompt + response) lengths.
        response_lengths: Per-sample response lengths.
        with_entropy:     Also return per-token entropy.
        max_seq_lens:     Per-sample padded lengths (bshd only).
        output_layer:     LM head module.
        output_weight:    Shared embedding weight or ``None``.

    Returns:
        ``{"log_probs": [...], "entropy": [...]}`` — lists of per-sample
        tensors with shape ``[R_i]``.
    """
    from .cp_utils import get_logits_and_tokens_offset_with_cp

    qkv_format = args.qkv_format
    chunk_size = args.log_probs_chunk_size
    tp_size = parallel_state.tp.size
    cp_size = parallel_state.cp.size
    sp_enabled = getattr(args, "sequence_parallel", False) and tp_size > 1

    # ---- reshape ----
    if qkv_format == "thd":
        hs = hidden_states.squeeze(1)
    else:
        hs = hidden_states.transpose(0, 1).contiguous().view(-1, hidden_states.size(-1))

    # ---- SP: gather sharded hidden states back to full sequence ----
    # With SP each TP rank holds [T/TP, H].  The per-sample offset logic
    # expects the full [T, H] layout, so we all-gather first and
    # temporarily disable output_layer.sequence_parallel (it would
    # otherwise try to reduce-scatter internally).
    old_sp_flag = getattr(output_layer, "sequence_parallel", None)
    if sp_enabled:
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        seq_len_before = hs.size(0)
        hs = gather_from_sequence_parallel_region(
            hs,
            tensor_parallel_output_grad=True,
            group=parallel_state.tp.group,
        )
        assert hs.size(0) == seq_len_before * tp_size, (
            f"SP gather: expected {seq_len_before * tp_size} tokens, " f"got {hs.size(0)}"
        )
        if old_sp_flag is not None:
            output_layer.sequence_parallel = False

    try:
        log_probs_list: list[torch.Tensor] = []
        entropy_list: list[torch.Tensor | None] = []

        # ---- per-sample loop ----
        end = 0
        for i, (tokens, total_length, response_length) in enumerate(
            zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

            # ---- slice hidden states for this sample's response ----
            if cp_size == 1:
                if qkv_format == "bshd":
                    e = max_seq_len * i + total_length
                    s = e - response_length
                else:
                    end += total_length
                    e = end
                    s = e - response_length

                hs_response = hs[s - 1 : e - 1]
                tokens_response = tokens[-response_length:]
            else:
                cp_chunk_sz, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                    total_length,
                    response_length,
                    qkv_format,
                    max_seq_len,
                )

                hs_0 = hs[end : end + cp_chunk_sz]
                hs_1 = hs[end + cp_chunk_sz : end + 2 * cp_chunk_sz]
                end += 2 * cp_chunk_sz

                hs_0 = hs_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
                tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

                hs_1 = hs_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
                tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

                assert hs_0.size(0) == tokens_0.size(0), f"CP chunk 0: hs {hs_0.size(0)} vs tokens {tokens_0.size(0)}"
                assert hs_1.size(0) == tokens_1.size(0), f"CP chunk 1: hs {hs_1.size(0)} vs tokens {tokens_1.size(0)}"

                hs_response = torch.cat([hs_0, hs_1], dim=0)
                tokens_response = torch.cat([tokens_0, tokens_1], dim=0)

            # ---- compute log-probs for this sample ----
            if hs_response.size(0) == 0:
                log_probs_list.append(hs_response.new_zeros((0,)))
                entropy_list.append(hs_response.new_zeros((0,)) if with_entropy else None)
                continue

            lp, ent = chunked_log_probs(
                hs_response,
                tokens_response,
                output_layer=output_layer,
                output_weight=output_weight,
                chunk_size=chunk_size,
                temperature=args.rollout_temperature,
                tp_size=tp_size,
                tp_group=parallel_state.tp.group,
                with_entropy=with_entropy,
                true_on_policy=getattr(args, "true_on_policy_mode", False),
            )

            log_probs_list.append(lp)
            entropy_list.append(ent)
    finally:
        if sp_enabled and old_sp_flag is not None:
            output_layer.sequence_parallel = old_sp_flag

    result: dict[str, list] = {"log_probs": log_probs_list}
    if with_entropy:
        result["entropy"] = entropy_list
    return result
