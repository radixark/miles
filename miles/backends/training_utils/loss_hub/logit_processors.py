from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch

from miles.backends.training_utils.cp_utils import allgather_cp_redistribute, get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.sampling_mask import build_local_sampling_mask
from miles.utils.sampling_mask import RolloutSamplingMask


def _iter_response_chunk_parts(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    include_response_indices: bool,
) -> Iterator[tuple[tuple[torch.Tensor, torch.Tensor, Sequence[int]], ...]]:
    """Yield contiguous response-logit parts for each sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        A tuple of contiguous `(logits_chunk, tokens_chunk, response_indices)`
        parts for one sample. Zigzag context parallelism yields its two local
        sequence parts separately so callers can process them without
        materializing a full-response logits concatenation.
    """
    if not args.true_on_policy_mode:
        # FSDP hands native bf16 here (no full-vocab fp32 buffer); chunks are upcast to fp32 downstream
        assert logits.dtype in (torch.float32, torch.bfloat16), f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if logits.size(-1) > 1 and args.rollout_temperature > 0 and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)
    if args.true_on_policy_mode:
        if getattr(args, "bf16", False):
            logits = logits.to(torch.bfloat16)
        elif getattr(args, "fp16", False):
            logits = logits.to(torch.float16)

    yield from _iter_response_tensor_parts(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=include_response_indices,
    )


def _iter_response_tensor_parts(
    tensor: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    include_response_indices: bool,
) -> Iterator[tuple[tuple[torch.Tensor, torch.Tensor, Sequence[int]], ...]]:
    """Yield response-aligned parts from a sequence tensor without changing its values.

    This is shared by ordinary logits processing and the checkpointed SFT
    output projection, which applies the same CP/token alignment directly to
    decoder hidden states before creating logits.
    """

    qkv_format = args.qkv_format
    if tensor.ndim != 3:
        raise ValueError(f"tensor must have shape [B, S, D], got {tensor.shape}")
    if qkv_format == "thd":
        if tensor.size(0) != 1:
            raise ValueError(f"THD tensor must have batch dimension 1, got {tensor.shape}")
        tensor = tensor.squeeze(0)
    else:
        if max_seq_lens is None:
            raise ValueError("max_seq_lens is required for qkv_format='bshd'")
        tensor = tensor.reshape(-1, tensor.size(-1))

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
                logits_chunk = tensor[start - 1 : end - 1]
            else:
                end += total_length
                start = end - response_length
                logits_chunk = tensor[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:] if response_length else tokens[0:0]
            response_indices = range(response_length) if include_response_indices else ()
            response_chunks = ((logits_chunk, tokens_chunk, response_indices),)
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = tensor.size(0)
            cp_rank = parallel_state.cp.rank
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = tensor[0:0]
                tokens_chunk = tokens[0:0]
                response_indices = ()
            else:
                logits_chunk = tensor[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
                response_indices = (
                    range(
                        s - logit_global_start,
                        e - logit_global_start,
                    )
                    if include_response_indices
                    else ()
                )
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
            response_chunks = ((logits_chunk, tokens_chunk, response_indices),)
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = tensor[end : end + chunk_size], tensor[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            if include_response_indices:
                prompt_length = total_length - response_length
                response_indices_0: Sequence[int] = range(
                    tokens_offset[0][0] - prompt_length,
                    tokens_offset[0][1] - prompt_length,
                )
                response_indices_1: Sequence[int] = range(
                    tokens_offset[1][0] - prompt_length,
                    tokens_offset[1][1] - prompt_length,
                )
            else:
                response_indices_0 = ()
                response_indices_1 = ()
            response_chunks = (
                (logits_0, tokens_0, response_indices_0),
                (logits_1, tokens_1, response_indices_1),
            )

        seq_start += total_length

        if include_response_indices:
            for _logits_chunk, tokens_chunk, response_indices in response_chunks:
                assert len(response_indices) == tokens_chunk.size(0)
        yield response_chunks


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
    """Yield one response-aligned tensor per sample for legacy consumers."""
    response_chunk_parts = _iter_response_chunk_parts(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=include_response_indices,
    )
    for chunks in response_chunk_parts:
        if len(chunks) == 1:
            yield chunks[0]
            continue

        logits_chunks, tokens_chunks, index_chunks = zip(*chunks, strict=True)
        response_indices = tuple(index for indices in index_chunks for index in indices)
        yield torch.cat(logits_chunks, dim=0), torch.cat(tokens_chunks, dim=0), response_indices


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
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
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
    parallel_state = get_parallel_state()
    log_probs_list = []
    entropy_list = []
    response_chunk_parts = _iter_response_chunk_parts(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=rollout_sampling_mask is not None,
    )
    for sample_index, response_chunks in enumerate(response_chunk_parts):
        sample_log_probs = []
        sample_entropies = []
        for logits_chunk, tokens_chunk, response_indices in response_chunks:
            sampling_mask = None
            if rollout_sampling_mask is not None:
                sampling_mask = build_local_sampling_mask(
                    logits_chunk,
                    rollout_sampling_mask[sample_index],
                    response_indices,
                    tp_rank=parallel_state.tp.rank,
                )
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
            )
            sample_log_probs.append(log_prob.squeeze(-1))
            if with_entropy:
                assert entropy is not None
                sample_entropies.append(entropy)

        log_probs_list.append(sample_log_probs[0] if len(sample_log_probs) == 1 else torch.cat(sample_log_probs))
        if with_entropy:
            entropy_list.append(sample_entropies[0] if len(sample_entropies) == 1 else torch.cat(sample_entropies))

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list

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
