"""Record the sampler's top-k candidate log-probs at generation time."""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import numpy as np
import pybase64

from miles.rollout.generate_utils.sampling_mask import decode_flat_sampling_mask
from miles.utils.rollout_topk_logprobs import (
    requests_support_logprobs,
    rollout_topk_from_support,
    validate_rollout_topk_logprobs_sampling,
)
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
    if requests_support_logprobs(args):
        # SGLang's support mode returns the actual post-filter behavior distribution.
        for key in (
            "top_logprobs",
            "top_logprobs_num",
            "return_flat_raw_output_top_logprobs",
            "return_flat_raw_top_logprobs_b64",
        ):
            request.pop(key, None)
        request["sampling_logprobs_mode"] = "support"
    else:
        request.pop("sampling_logprobs_mode", None)
        request["top_logprobs" if openai else "top_logprobs_num"] = k
        # base64 [tokens, k] arrays: no per-candidate detokenization, and -inf survives (JSON would send null)
        request["return_flat_raw_output_top_logprobs"] = True
        request["return_flat_raw_top_logprobs_b64"] = True


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
    ids = np.full((n, k), -1, dtype=np.int32)
    logps = np.full((n, k), -np.inf, dtype=np.float32)
    # A zero-token turn keeps its empty [0, k] arrays whatever SGLang reports.
    if n and sampling_logprobs_mode == "support":
        support_ids, lengths, support_logps = decode_flat_sampling_mask(meta)
        if len(lengths) != n:
            raise ValueError(
                "Rollout top-k logprobs collection requires SGLang's sampling support for every generated token"
            )
        ids, logps = rollout_topk_from_support(
            ids=support_ids,
            offsets=np.concatenate(([0], np.cumsum(lengths))),
            logprobs=support_logps,
            forced=np.zeros(n, dtype=bool),
            k=k,
        )
    elif n:
        flat_ids, flat_logps = _decode_flat_output_top_logprobs(meta)
        if len(flat_ids) != n:
            raise ValueError(
                "Rollout top-k logprobs collection requires SGLang output_top_logprobs for every generated token"
            )
        if flat_ids.shape[1] < k:
            # e.g. a router capping top_logprobs; padding would silently shrink the candidate set
            raise ValueError(
                f"SGLang returned {flat_ids.shape[1]} output top logprobs per token, "
                f"fewer than --rollout-top-logprobs-num {k}"
            )
        ids[:] = flat_ids[:, :k]
        logps[:] = flat_logps[:, :k]
    for name, values in (("rollout_topk_token_ids", ids), ("rollout_topk_log_probs", logps)):
        previous = getattr(
            sample, name
        )  # config-access-exempt: sample field name is selected from the paired top-k array fields
        setattr(sample, name, values if previous is None else np.concatenate((previous, values)))


def _decode_flat_output_top_logprobs(meta: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Candidate ids and logprobs `[tokens, width]` from SGLang's base64 `return_flat_raw_output_top_logprobs`."""
    if "output_top_logprobs_val_flat_b64" not in meta:
        raise ValueError(
            "Rollout top-k logprobs collection requires SGLang's base64 flat output top logprobs "
            "(return_flat_raw_output_top_logprobs), behind a router that forwards it"
        )
    if meta["output_top_logprobs_null_prefix"]:
        raise ValueError(
            "Rollout top-k logprobs collection requires SGLang output_top_logprobs for every generated token"
        )
    shape = meta["output_top_logprobs_shape"]

    def decode(field: str, dtype: type) -> np.ndarray:
        # SGLang writes contiguous little-endian bytes; the dtype marker names only the width
        wire_dtype = np.dtype(meta[f"{field}_flat_b64_dtype"]).newbyteorder("<")
        raw = pybase64.b64decode(meta[f"{field}_flat_b64"])
        return np.frombuffer(raw, dtype=wire_dtype).astype(dtype, copy=False).reshape(shape)

    return decode("output_top_logprobs_idx", np.int32), decode("output_top_logprobs_val", np.float32)


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
        values = getattr(
            sample, field
        )  # config-access-exempt: sample field name is selected from the explicit rollout probability fields
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
    a, b = getattr(first, field), getattr(
        second, field
    )  # config-access-exempt: sample field name is selected from the explicit rollout probability fields
    if a is None and b is None:
        return None
    if a is None or b is None or a.shape[1] != b.shape[1]:
        raise ValueError(f"Both turns must carry matching {field} for rollout top-k logprobs collection")
    fill = -1 if field == "rollout_topk_token_ids" else -np.inf
    return np.concatenate((a, np.full((gap, a.shape[1]), fill, dtype=a.dtype), b))
