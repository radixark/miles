from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F

from miles.backends.training_utils.data.context_parallel import (
    LocalResponseRows,
    allgather_cp_redistribute,
    iter_local_response_rows,
)
from miles.backends.training_utils.data.sampling_mask import build_local_sampling_mask, local_support_ids
from miles.backends.training_utils.loss.hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.score_centering import selected_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.sampling_mask import RolloutSamplingMask


def _flatten_logits(logits: torch.Tensor, qkv_format: str, max_seq_lens: list[int] | None) -> torch.Tensor:
    """``[1, T, V]`` (thd) or ``[B, S, V]`` (bshd) model logits as ``[rows, V]``, a view."""
    assert len(logits.shape) == 3, f"{logits.shape}"
    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        return logits.squeeze(0)
    assert max_seq_lens is not None
    return logits.view(-1, logits.size(-1))


def _iter_response_chunks(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    include_response_indices: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, Sequence[int]]]:
    """Yield response logits, tokens, and original response indices per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample (``context_parallel.iter_local_response_rows`` owns the row layout).

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk, response_indices)`, where
        `logits_chunk` is shape `[R, V]` (policy) or `[R, 1]` (value), and
        `tokens_chunk` is shape `[R]` (1D int64). `response_indices` maps every
        local row back to the full response. The mapping is empty when
        `include_response_indices` is false.
    """
    if not args.true_on_policy_mode:
        # Model-precision callers hand native bf16/fp16 logits; chunks are upcast to fp32 downstream
        assert logits.dtype in (torch.float32, torch.bfloat16, torch.float16), f"{logits.dtype}"
    logits = _flatten_logits(logits, args.qkv_format, max_seq_lens)

    if args.true_on_policy_mode:
        if logits.size(-1) > 1 and args.rollout_temperature > 0 and args.rollout_temperature != 1.0:
            logits = logits.div(args.rollout_temperature)
        if getattr(args, "bf16", False):
            logits = logits.to(torch.bfloat16)
        elif getattr(args, "fp16", False):
            logits = logits.to(torch.float16)

    layout = iter_local_response_rows(
        logits.size(0),
        qkv_format=args.qkv_format,
        allgather_cp=args.allgather_cp,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    for sample_rows, tokens, total_length, response_length in zip(
        layout, unconcat_tokens, total_lengths, response_lengths, strict=False
    ):
        pieces = [logits[row_start:row_end] for row_start, row_end in sample_rows.row_ranges]
        logits_chunk = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        tokens_chunk = sample_rows.tokens(tokens, total_length - response_length)
        yield logits_chunk, tokens_chunk, (sample_rows.response_indices() if include_response_indices else ())


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample."""
    for logits_chunk, tokens_chunk, _ in _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=False,
    ):
        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    entropy_requires_grad: bool = True,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    rollout_sampling_mask: Sequence[RolloutSamplingMask] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy is computed and returned only
    when requested. With `--log-probs-backend fused` all samples go through one
    fused op, normalized over the true vocabulary (`args.vocab_size`).

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `calculate_log_probs_and_entropy`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        entropy_requires_grad: If False, compute entropy as an observed metric
            without attaching it to the autograd graph.
        non_loss_data: Unused; kept for API compatibility.
        rollout_sampling_mask: One ``RolloutSamplingMask`` per sample,
            covering every response token.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    if rollout_sampling_mask is not None:
        for sample_index, (sampling_mask, response_length) in enumerate(
            zip(rollout_sampling_mask, response_lengths, strict=True)
        ):
            if len(sampling_mask) != response_length:
                raise ValueError(
                    f"sampling-mask length {len(sampling_mask)} != response length "
                    f"{response_length} for sample {sample_index}"
                )
    if getattr(args, "log_probs_backend", "torch") == "fused":
        res = _fused_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            max_seq_lens=max_seq_lens,
            rollout_sampling_mask=rollout_sampling_mask,
        )
    else:
        res = _torch_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            max_seq_lens=max_seq_lens,
            rollout_sampling_mask=rollout_sampling_mask,
        )

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def _torch_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool,
    entropy_requires_grad: bool,
    max_seq_lens: list[int] | None,
    rollout_sampling_mask: Sequence[RolloutSamplingMask] | None,
) -> dict[str, list[torch.Tensor]]:
    """Each sample's response chunk through ``calculate_log_probs_and_entropy``."""
    parallel_state = get_parallel_state()
    log_probs_list = []
    entropy_list = []
    response_chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=rollout_sampling_mask is not None,
    )
    for sample_index, (logits_chunk, tokens_chunk, response_indices) in enumerate(response_chunks):
        sampling_mask = None
        if rollout_sampling_mask is not None:
            sampling_mask = build_local_sampling_mask(
                logits_chunk,
                rollout_sampling_mask[sample_index],
                response_indices,
                tp_rank=parallel_state.tp.rank,
            )
        if getattr(args, "loss_type", None) == "score_centering" and sampling_mask is None:
            # Reference KL compares these with the score-centering actor score, which
            # normalizes over the true vocabulary instead of Megatron's padded one.
            log_prob, entropy = selected_log_probs_and_entropy(
                logits_chunk,
                tokens_chunk.unsqueeze(-1),
                group=parallel_state.tp.group if parallel_state.tp.size > 1 else None,
                vocab_size=getattr(args, "vocab_size", None),
                temperature=args.rollout_temperature,
                chunk_size=args.log_probs_chunk_size,
                with_entropy=with_entropy,
            )
            if not entropy_requires_grad:
                entropy = entropy.detach()
        else:
            log_prob, entropy = calculate_log_probs_and_entropy(
                logits_chunk,
                tokens_chunk,
                parallel_state.tp.group,
                with_entropy=with_entropy,
                entropy_requires_grad=entropy_requires_grad,
                chunk_size=args.log_probs_chunk_size,
                true_on_policy=args.true_on_policy_mode,
                vocab_size=getattr(args, "vocab_size", None),
                sampling_mask=sampling_mask,
                temperature=1.0 if args.true_on_policy_mode else args.rollout_temperature,
                debug_unified_grad_fused_logprob=args.debug_unified_grad_fused_logprob,
            )

        log_probs_list.append(log_prob.squeeze(-1))
        if with_entropy:
            entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    return res


def _fused_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool,
    entropy_requires_grad: bool,
    max_seq_lens: list[int] | None,
    rollout_sampling_mask: Sequence[RolloutSamplingMask] | None,
) -> dict[str, list[torch.Tensor]]:
    """Each response row's next token through ``fused_response_log_probs``.

    With a rollout sampling mask the log-probs are over each token's sampling support, and the
    entropy stays over the vocabulary, as on the torch path.
    """
    sample_support = None
    if rollout_sampling_mask is not None:
        sample_support = lambda i, sample_rows: local_support_ids(  # noqa: E731
            rollout_sampling_mask[i], sample_rows.response_indices(), logits.device
        )
    scores = fused_response_log_probs(
        logits,
        args=args,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        sample_targets=lambda i, sample_rows: sample_rows.tokens(
            unconcat_tokens[i], total_lengths[i] - response_lengths[i]
        ),
        sample_support=sample_support,
        with_entropy=with_entropy,
        entropy_requires_grad=entropy_requires_grad,
    )
    res = {"log_probs": scores.log_probs}
    if with_entropy:
        res["entropy"] = scores.entropy
    return res


class FusedResponseScores(NamedTuple):
    """Per-sample outputs of ``fused_response_log_probs``."""

    log_probs: list[torch.Tensor]  # of each sample's targets, in their shape
    entropy: list[torch.Tensor] | None
    support_log_probs: list[torch.Tensor] | None  # [R_i, S] for the support ids, with a support


def fused_response_log_probs(
    logits: torch.Tensor,
    *,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
    sample_targets: Callable[[int, LocalResponseRows], torch.Tensor],
    sample_support: Callable[[int, LocalResponseRows], torch.Tensor] | None = None,
    with_entropy: bool,
    entropy_over_support: bool = False,
    entropy_requires_grad: bool = True,
) -> FusedResponseScores:
    """Log-probs of ``sample_targets(i, rows)`` (``[R_i]`` or ``[R_i, K]``) and the entropy at this rank's
    response rows, all samples in one fused call.

    With ``sample_support`` (``[R_i, S_i]`` ids, ``-1`` padding) the log-probs are normalized over
    each row's support and the support ids are scored too; the entropy is over the vocabulary unless
    ``entropy_over_support``.
    """
    # imported here: the op needs triton, which not every host that imports the losses has
    from miles.backends.training_utils.loss.hub.fused_log_probs import (
        fused_log_probs_and_entropy,
        fused_support_log_probs,
    )

    flat_logits = _flatten_logits(logits, args.qkv_format, max_seq_lens)
    layout = iter_local_response_rows(
        flat_logits.size(0),
        qkv_format=args.qkv_format,
        allgather_cp=args.allgather_cp,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    device = flat_logits.device
    row_ranges, targets, supports = [], [], []
    for i, sample_rows in enumerate(layout):
        row_ranges.extend(sample_rows.row_ranges)
        targets.append(sample_targets(i, sample_rows).to(device))
        if sample_support is not None:
            supports.append(sample_support(i, sample_rows).to(device))
    rows = torch.cat([torch.arange(row_start, row_end, device=device) for row_start, row_end in row_ranges])
    options = dict(
        tp_group=get_parallel_state().tp.group,
        vocab_size=getattr(args, "vocab_size", None),  # a Megatron flag; other backends' logits are unpadded
        temperature=args.rollout_temperature,
        with_entropy=with_entropy,
        entropy_requires_grad=entropy_requires_grad,
        # the checkpointed loss replays its forward on these logits during backward
        inplace_backward=not args.recompute_loss_function,
    )
    lengths = [sample.size(0) for sample in targets]
    support_log_probs = None
    if sample_support is None:
        log_probs, entropy = fused_log_probs_and_entropy(flat_logits, rows, torch.cat(targets), **options)
    else:
        width = max((support.size(1) for support in supports), default=0)
        support = torch.cat([F.pad(support, (0, width - support.size(1)), value=-1) for support in supports])
        log_probs, all_support_log_probs, entropy = fused_support_log_probs(
            flat_logits, rows, torch.cat(targets), support, entropy_over_support=entropy_over_support, **options
        )
        support_log_probs = list(all_support_log_probs.split(lengths))
    return FusedResponseScores(
        list(log_probs.split(lengths)), list(entropy.split(lengths)) if with_entropy else None, support_log_probs
    )


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        # upcast (no-op for fp32) so value-head outputs stay fp32 even when logits arrive bf16
        value_list.append(logits_chunk.squeeze(-1).float())

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res
