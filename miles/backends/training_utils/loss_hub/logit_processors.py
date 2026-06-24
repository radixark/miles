from argparse import Namespace
from collections.abc import Iterator

import torch

from miles.backends.training_utils.cp_utils import allgather_cp_redistribute, get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

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
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    if not args.true_on_policy_mode:
        assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if logits.size(-1) > 1 and args.rollout_temperature > 0 and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)
    if args.true_on_policy_mode:
        if getattr(args, "bf16", False):
            logits = logits.to(torch.bfloat16)
        elif getattr(args, "fp16", False):
            logits = logits.to(torch.float16)

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
                logits_chunk = logits[start - 1 : end - 1]
            else:
                end += total_length
                start = end - response_length
                logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
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
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def build_shifted_tokens_bshd(
    num_tokens: int,
    device: torch.device,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    max_seq_lens: list[int],
) -> torch.Tensor:
    full_tokens = torch.zeros(num_tokens, dtype=torch.long, device=device)
    for i, (tokens, total_length) in enumerate(zip(unconcat_tokens, total_lengths, strict=False)):
        seq_start = max_seq_lens[i] * i
        full_tokens[seq_start : seq_start + total_length - 1] = tokens[1:total_length]
    return full_tokens


def extract_per_sample_bshd(
    log_prob_full: torch.Tensor,
    entropy_full: torch.Tensor | None,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    log_probs_list = []
    entropy_list = []
    for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
        end = max_seq_lens[i] * i + total_length
        start = end - response_length
        log_probs_list.append(log_prob_full[start - 1 : end - 1])
        entropy_list.append(entropy_full[start - 1 : end - 1] if entropy_full is not None else None)
    return log_probs_list, entropy_list


def get_log_probs_and_entropy_from_hidden_states(
    hidden_states: torch.Tensor,
    *,
    projection,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    parallel_state = get_parallel_state()
    hidden_states = projection.gather_sp(hidden_states)
    hidden_states = hidden_states.contiguous().view(-1, hidden_states.size(-1)).contiguous()
    num_tokens = hidden_states.size(0)
    tp_group = parallel_state.tp.group
    seq_chunk_size = args.chunked_tp_logprob_seq_chunk_size
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)

    full_tokens = build_shifted_tokens_bshd(
        num_tokens, hidden_states.device, unconcat_tokens, total_lengths, max_seq_lens
    )

    log_prob_chunks = []
    entropy_chunks = []
    for start in range(0, num_tokens, seq_chunk_size):
        end = min(start + seq_chunk_size, num_tokens)
        logits_chunk = projection.linear(hidden_states[start:end]).float().contiguous()
        if rollout_temperature != 1.0:
            logits_chunk = logits_chunk / rollout_temperature
        log_prob_chunk, entropy_chunk = calculate_log_probs_and_entropy(
            logits_chunk,
            full_tokens[start:end],
            tp_group,
            with_entropy=with_entropy,
            chunk_size=-1,
            true_on_policy=False,
            vocab_size=getattr(args, "vocab_size", None),
            need_entropy_grad=with_entropy,
        )
        log_prob_chunks.append(log_prob_chunk.reshape(-1))
        if with_entropy:
            entropy_chunks.append(entropy_chunk.reshape(-1))

    log_prob_full = torch.cat(log_prob_chunks, dim=0)
    entropy_full = torch.cat(entropy_chunks, dim=0) if with_entropy else None

    log_probs_list, entropy_list = extract_per_sample_bshd(
        log_prob_full, entropy_full, total_lengths, response_lengths, max_seq_lens
    )
    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list
    return res


def get_log_probs_and_entropy(
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
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    projection = getattr(args, "actor_projection", None)
    if projection is not None and projection.bypass_enabled:
        return get_log_probs_and_entropy_from_hidden_states(
            logits,
            projection=projection,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            max_seq_lens=max_seq_lens,
        )

    parallel_state = get_parallel_state()
    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            parallel_state.tp.group,
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
            true_on_policy=args.true_on_policy_mode,
            vocab_size=getattr(args, "vocab_size", None),
            need_entropy_grad=with_entropy,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

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
        value_list.append(logits_chunk.squeeze(-1))

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
