"""Collect the sampler's distribution at generation time for score centering."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import numpy as np

from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.score_centering import score_centering_top_k, validate_score_centering_sampling
from miles.utils.types import Sample


def configure_score_centering_request(args: Namespace, request: dict[str, Any], *, openai: bool = False) -> None:
    """Request candidate probabilities, validating the actual per-call settings."""
    k = score_centering_top_k(args)
    if not k:
        return
    sampling = request if openai else request["sampling_params"]
    # Explicit defaults avoid model generation_config changing the distribution.
    for key, default in (("temperature", args.rollout_temperature), ("top_p", 1.0), ("top_k", -1), ("min_p", 0.0)):
        if sampling.get(key) is None:
            sampling[key] = default
    validate_score_centering_sampling(sampling, temperature=args.rollout_temperature, candidate_count=k)
    if sampling["top_p"] < 1.0 or sampling["top_k"] > 0:
        # SGLang's support mode returns the actual post-filter behavior distribution.
        request.pop("top_logprobs", None)
        request.pop("top_logprobs_num", None)
        request["sampling_logprobs_mode"] = "support"
    elif openai:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs"] = k
    else:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs_num"] = max(k, request.get("top_logprobs_num", 0) or 0)


def append_score_centering_topk(
    sample: Sample, meta: Mapping[str, Any], k: int, *, sampling_logprobs_mode: str = "selected"
) -> None:
    """Append compact [response, k] arrays after appending generated tokens.

    No rescoring is allowed: with stale rollouts it would replace the behavior
    policy. Slots unavailable from the server are represented by ID -1/logp -inf.
    """
    if not k:
        return
    n = len(meta.get("output_token_logprobs") or [])
    support_mode = sampling_logprobs_mode == "support"
    if sampling_logprobs_mode not in ("selected", "support"):
        raise ValueError(f"Unsupported score-centering sampling logprobs mode: {sampling_logprobs_mode}")
    support_ids = meta.get("output_token_sampling_mask") if support_mode else None
    rows = meta.get("output_token_sampling_logprobs" if support_mode else "output_top_logprobs")
    if support_mode and n and (sample.rollout_sampling_mask is None or support_ids is None):
        raise ValueError("Score centering requires SGLang sampling support IDs")
    if rows is None and n:
        field = "output_token_sampling_logprobs" if support_mode else "output_top_logprobs"
        raise ValueError(f"Score centering requires SGLang {field} from generation")
    rows = rows if rows is not None else []
    if len(rows) != n or (support_mode and len(support_ids) != n):
        raise ValueError("Score-centering candidate rows do not match generated tokens")
    ids = np.full((n, k), -1, dtype=np.int32)
    logps = np.full((n, k), -np.inf, dtype=np.float32)
    for i, entries in enumerate(rows):
        if not entries:
            raise ValueError(f"Missing score-centering candidates at generated position {i}")
        if support_mode:
            token_ids = support_ids[i]
            if len(entries) != len(token_ids) or len(entries) > k:
                raise ValueError("Score-centering sampling support exceeds or disagrees with saved candidates")
            probabilities = np.asarray(entries, dtype=np.float64)
        else:
            entries = entries[:k]
            token_ids = [entry[1] for entry in entries]
            probabilities = np.asarray([entry[0] for entry in entries], dtype=np.float64)
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 for token_id in token_ids):
            raise ValueError("Score-centering candidates must have non-negative integer token IDs")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("Duplicate score-centering candidate token IDs")
        mass = np.exp(probabilities).sum()
        if np.isnan(probabilities).any() or (probabilities > 0).any() or mass > 1 + 1e-5:
            raise ValueError("Invalid score-centering candidate probabilities")
        if support_mode and (not np.isfinite(probabilities).all() or not np.isclose(mass, 1.0, atol=1e-5)):
            raise ValueError("Score-centering sampling support probabilities must sum to one")
        ids[i, : len(entries)] = token_ids
        logps[i, : len(entries)] = probabilities
    prefix_length = sample.response_length - n
    if support_mode and n:
        in_support, lengths = _candidate_support_membership(ids, sample.rollout_sampling_mask, prefix_length)
        if not np.array_equal(in_support, ids >= 0) or not np.array_equal(in_support.sum(-1), lengths):
            raise ValueError("Score-centering candidates do not match the sampling support")
    for field, values in (("rollout_topk_token_ids", ids), ("rollout_topk_log_probs", logps)):
        previous = getattr(sample, field)
        if previous is None:
            if prefix_length:
                raise ValueError("Cannot start collecting score-centering candidates midway through a response")
            setattr(sample, field, values)
        else:
            if previous.shape != (prefix_length, k):
                raise ValueError(f"Misaligned {field}: {previous.shape}, expected {(prefix_length, k)}")
            setattr(sample, field, np.concatenate((previous, values)))


def _candidate_support_membership(
    ids: np.ndarray, support: RolloutSamplingMask, start: int
) -> tuple[np.ndarray, np.ndarray]:
    """Match candidate IDs to ragged support IDs without a per-token Python loop."""
    n = len(ids)
    if start < 0 or start + n > len(support):
        raise ValueError("Score-centering candidates and sampling support are misaligned")
    if not n:
        return np.zeros_like(ids, dtype=bool), np.empty(0, dtype=np.int64)
    flat_ids, offsets = support._as_tensors()
    flat_ids = flat_ids.numpy()
    offsets = offsets.numpy()
    lengths = np.diff(offsets[start : start + n + 1])
    support_ids = flat_ids[offsets[start] : offsets[start + n]]
    stride = 1 << 32  # Token IDs are nonnegative int32; rows remain distinct.
    support_rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
    support_keys = np.sort(support_rows * stride + support_ids.astype(np.int64))
    if (np.diff(support_keys) == 0).any():
        raise ValueError("Sampling support contains duplicate token IDs")
    candidate_keys = np.arange(n, dtype=np.int64)[:, None] * stride + ids.astype(np.int64)
    positions = np.searchsorted(support_keys, candidate_keys)
    membership = (ids >= 0) & (support_keys[np.minimum(positions, len(support_keys) - 1)] == candidate_keys)
    return membership, lengths


def append_score_centering_observations(sample: Sample, count: int) -> None:
    """Pad non-trained tool/observation positions before extending the response."""
    for field, fill in (("rollout_topk_token_ids", -1), ("rollout_topk_log_probs", -np.inf)):
        values = getattr(sample, field)
        if values is not None:
            padding = np.full((count, values.shape[1]), fill, dtype=values.dtype)
            setattr(sample, field, np.concatenate((values, padding)))


def validate_score_centering_sample(sample: Sample, k: int) -> None:
    """Validate the contract also for custom producers and restored rollouts."""
    sample.validate()
    ids, logps = sample.rollout_topk_token_ids, sample.rollout_topk_log_probs
    if ids is None or logps is None or sample.rollout_log_probs is None:
        raise ValueError("Score centering requires candidate and sampled-token logprobs on every rollout")
    if ids.shape != (sample.response_length, k):
        raise ValueError("Score-centering candidates must have shape [response_length, score_centering_top_k]")
    if ids.dtype != np.int32 or logps.dtype != np.float32:
        raise ValueError("Score-centering candidates require int32 IDs and float32 logprobs")
    valid = ids >= 0
    active = np.asarray(sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length, dtype=bool)
    if (ids < -1).any() or (active & ~valid.any(-1)).any():
        raise ValueError("Every trained token needs score-centering candidates")
    if (active & ~np.isfinite(logps).any(-1)).any():
        raise ValueError("Every trained token needs positive candidate probability mass")
    if np.isnan(logps).any() or (logps > 0).any() or (~np.isneginf(logps[~valid])).any():
        raise ValueError("Invalid score-centering logprobs or candidate padding")
    if (np.exp(logps.astype(np.float64)).sum(-1) > 1 + 1e-5).any():
        raise ValueError("Score-centering candidate probability mass exceeds one")
    # Batched sorting avoids one Python/NumPy call per generated token. Repeated
    # -1 padding is allowed; nonnegative candidate IDs must be unique per row.
    sorted_ids = np.sort(ids, axis=-1)
    if ((sorted_ids[:, 1:] >= 0) & (sorted_ids[:, 1:] == sorted_ids[:, :-1])).any():
        raise ValueError("Duplicate score-centering candidate token IDs")
    if sample.rollout_sampling_mask is not None:
        in_support, lengths = _candidate_support_membership(ids, sample.rollout_sampling_mask, 0)
        if ((valid != in_support) & active[:, None]).any() or ((valid.sum(-1) != lengths) & active).any():
            raise ValueError("Score-centering candidates must equal the sampling support")
        head_mass = np.exp(logps.astype(np.float64)).sum(-1)
        if not np.allclose(head_mass[active], 1.0, atol=1e-5):
            raise ValueError("Filtered score-centering candidates must sum to one")
    sampled = np.asarray(sample.rollout_log_probs)
    if not np.isfinite(sampled[active]).all() or (sampled[active] > 0).any():
        raise ValueError("Invalid score-centering sampled-token logprobs")
    tokens = np.asarray(sample.tokens[-sample.response_length :] if sample.response_length else [], dtype=np.int64)
    matches = (ids == tokens[:, None]) & valid & active[:, None]
    if not np.allclose(np.broadcast_to(sampled[:, None], logps.shape)[matches], logps[matches], atol=1e-5, rtol=1e-5):
        raise ValueError("Sampled and candidate probabilities must come from the same sampler distribution")


def merge_score_centering_field(first: Sample, second: Sample, field: str, gap: int) -> np.ndarray | None:
    a, b = getattr(first, field), getattr(second, field)
    if a is None and b is None:
        return None
    if a is None or b is None or a.shape[1] != b.shape[1]:
        raise ValueError(f"Both turns must carry matching {field} for score centering")
    fill = -1 if field == "rollout_topk_token_ids" else -np.inf
    return np.concatenate((a, np.full((gap, a.shape[1]), fill, dtype=a.dtype), b))
