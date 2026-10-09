"""Record the sampler's top-k candidate log-probs at generation time."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import numpy as np

from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sampling
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


def configure_rollout_topk_logprobs_request(args: Namespace, request: dict[str, Any], *, openai: bool = False) -> None:
    """Request candidate probabilities, validating the actual per-call settings.

    The rollout args own ``top_logprobs_num`` / ``top_logprobs`` and ``sampling_logprobs_mode``
    on training requests and silently override client-supplied values.
    """
    k = args.rollout_top_logprobs_num
    if not k:
        return
    sampling = request if openai else request["sampling_params"]
    # Explicit defaults avoid model generation_config changing the distribution.
    for key, default in (("temperature", args.rollout_temperature), ("top_p", 1.0), ("top_k", -1), ("min_p", 0.0)):
        if sampling.get(key) is None:
            sampling[key] = default
    validate_rollout_topk_logprobs_sampling(sampling, temperature=args.rollout_temperature, candidate_count=k)
    if args.rollout_sampling_logprobs_mode == "support":
        # SGLang's support mode returns the actual post-filter behavior distribution.
        request.pop("top_logprobs", None)
        request.pop("top_logprobs_num", None)
        request["sampling_logprobs_mode"] = "support"
    elif openai:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs"] = k
    else:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs_num"] = k


def append_rollout_topk_logprobs(
    sample: Sample, meta: Mapping[str, Any], k: int, *, sampling_logprobs_mode: str = "selected"
) -> None:
    """Append compact [response, k] arrays after appending generated tokens.

    No rescoring is allowed: with stale rollouts it would replace the behavior
    policy. Slots unavailable from the server are represented by ID -1/logp -inf.
    append_sampling_metadata validates the sampling support on the same response,
    and CI checks the whole sample with validate_rollout_topk_logprobs_sample.
    """
    if not k:
        return
    n = len(meta.get("output_token_logprobs") or [])
    support_mode = sampling_logprobs_mode == "support"
    field = "output_token_sampling_logprobs" if support_mode else "output_top_logprobs"
    rows = meta.get(field)
    if n and (rows is None or len(rows) != n):
        raise ValueError(f"Rollout top-k logprobs collection requires SGLang {field} for every generated token")
    ids = np.full((n, k), -1, dtype=np.int32)
    logps = np.full((n, k), -np.inf, dtype=np.float32)
    for i, entries in enumerate(rows or []):
        if support_mode:
            if len(entries) > k:
                raise ValueError("Rollout top-k logprobs sampling support exceeds --rollout-top-logprobs-num")
            token_ids, probabilities = meta["output_token_sampling_mask"][i], entries
        else:
            token_ids = [entry[1] for entry in entries[:k]]
            probabilities = [entry[0] for entry in entries[:k]]
        ids[i, : len(token_ids)] = token_ids
        logps[i, : len(token_ids)] = probabilities
    for name, values in (("rollout_topk_token_ids", ids), ("rollout_topk_log_probs", logps)):
        previous = getattr(sample, name)
        setattr(sample, name, values if previous is None else np.concatenate((previous, values)))


def _support_membership(ids: np.ndarray, support: RolloutSamplingMask) -> tuple[np.ndarray, np.ndarray]:
    """Match candidate IDs to ragged support IDs without a per-token Python loop."""
    n = len(ids)
    if not n:
        return np.zeros_like(ids, dtype=bool), np.empty(0, dtype=np.int64)
    flat_ids, offsets = support._as_tensors()
    flat_ids = flat_ids.numpy()
    offsets = offsets.numpy()
    lengths = np.diff(offsets[: n + 1])
    support_ids = flat_ids[: offsets[n]]
    stride = 1 << 32  # Token IDs are nonnegative int32; rows remain distinct.
    support_rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
    support_keys = np.sort(support_rows * stride + support_ids.astype(np.int64))
    candidate_keys = np.arange(n, dtype=np.int64)[:, None] * stride + ids.astype(np.int64)
    positions = np.searchsorted(support_keys, candidate_keys)
    membership = (ids >= 0) & (support_keys[np.minimum(positions, len(support_keys) - 1)] == candidate_keys)
    return membership, lengths


def pad_rollout_topk_logprobs(sample: Sample, count: int) -> None:
    """Pad non-trained tool/observation positions before extending the response."""
    for field, fill in (("rollout_topk_token_ids", -1), ("rollout_topk_log_probs", -np.inf)):
        values = getattr(sample, field)
        if values is not None:
            padding = np.full((count, values.shape[1]), fill, dtype=values.dtype)
            setattr(sample, field, np.concatenate((values, padding)))


def validate_rollout_topk_logprobs_sample(sample: Sample, k: int) -> None:
    """Validate candidate distributions, including custom producers and restored rollouts.

    Full-sample sorting and support matching have significant CPU and memory costs;
    training callers must enable this validation only under --ci-test.
    """
    sample.validate()
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    if ids is None or logps is None or sample.rollout_log_probs is None:
        raise ValueError(
            "Rollout top-k logprobs collection requires candidate and sampled-token logprobs on every rollout"
        )
    if ids.shape[1] != k:
        raise ValueError("Rollout top-k logprobs candidates must have --rollout-top-logprobs-num columns")
    valid = ids >= 0
    sorted_ids = np.sort(ids, axis=-1)
    if ((sorted_ids[:, 1:] >= 0) & (sorted_ids[:, 1:] == sorted_ids[:, :-1])).any():
        raise ValueError("Duplicate rollout top-k logprobs candidate token IDs")
    active = np.asarray(sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length, dtype=bool)
    if (active & ~valid.any(-1)).any():
        raise ValueError("Every trained token needs rollout top-k logprobs candidates")
    if sample.rollout_sampling_mask is not None:
        in_support, lengths = _support_membership(ids, sample.rollout_sampling_mask)
        if ((valid != in_support) & active[:, None]).any() or ((valid.sum(-1) != lengths) & active).any():
            raise ValueError("Rollout top-k logprobs candidates must equal the sampling support")
    sampled = np.asarray(sample.rollout_log_probs)
    tokens = np.asarray(sample.tokens[-sample.response_length :] if sample.response_length else [], dtype=np.int64)
    matches = (ids == tokens[:, None]) & valid & active[:, None]
    if not np.allclose(np.broadcast_to(sampled[:, None], logps.shape)[matches], logps[matches], atol=1e-5, rtol=1e-5):
        raise ValueError("Sampled and candidate probabilities must come from the same sampler distribution")


def merge_rollout_topk_logprobs_field(first: Sample, second: Sample, field: str, gap: int) -> np.ndarray | None:
    a, b = getattr(first, field), getattr(second, field)
    if a is None and b is None:
        return None
    if a is None or b is None or a.shape[1] != b.shape[1]:
        raise ValueError(f"Both turns must carry matching {field} for rollout top-k logprobs collection")
    fill = -1 if field == "rollout_topk_token_ids" else -np.inf
    return np.concatenate((a, np.full((gap, a.shape[1]), fill, dtype=a.dtype), b))
