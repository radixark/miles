from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pybase64
import torch

from miles.utils.output_store.plan import OUTPUT_STORE_REQUEST_KEY, StorePlanError
from miles.utils.sampling_mask import RolloutSamplingMask, validate_sampling_support
from miles.utils.types import Sample


def should_return_sampling_mask(
    args: Namespace,
    sampling_params: Mapping[str, object] | None = None,
    *,
    evaluation: bool = False,
) -> bool:
    """Validate whether a training request must return its realized sampling support."""
    if evaluation:
        return False

    params = sampling_params or {}
    return validate_sampling_support_request(
        params,
        replay_enabled=args.use_sampling_support_replay,
        expected_temperature=float(args.rollout_temperature),
    )


def validate_sampling_support_request(
    sampling_params: Mapping[str, object],
    *,
    replay_enabled: bool,
    expected_temperature: float | None,
) -> bool:
    """Validate one resolved training request against its sampling replay contract."""
    request_top_p = float(1.0 if sampling_params.get("top_p") is None else sampling_params["top_p"])
    if not 0.0 < request_top_p <= 1.0:
        raise ValueError(f"training request top_p must be in (0, 1], got {request_top_p}")

    if not replay_enabled:
        raw_top_k = sampling_params.get("top_k")
        request_top_k = -1 if raw_top_k is None else int(raw_top_k)
        if request_top_p < 1.0 or request_top_k > 0:
            raise ValueError("bounded training-request sampling requires bounded rollout sampling")
        return False

    missing_params = [name for name in ("top_p", "top_k", "temperature") if sampling_params.get(name) is None]
    if missing_params:
        raise ValueError(f"sampling-support replay requires explicit request parameters: {', '.join(missing_params)}")

    request_top_k = int(sampling_params["top_k"])
    if request_top_k <= 0:
        raise ValueError(f"training request top_k must be positive, got {request_top_k}")

    request_temperature = float(sampling_params["temperature"])
    if expected_temperature is None:
        raise ValueError("sampling-support replay requires a registered training temperature")
    if request_temperature != expected_temperature:
        raise ValueError(
            f"request temperature {request_temperature} does not match the training temperature "
            f"{expected_temperature}"
        )

    unsupported = {
        "frequency_penalty": (0, 0.0, None),
        "presence_penalty": (0, 0.0, None),
        "repetition_penalty": (1, 1.0, None),
        "logit_bias": ({}, None),
        "custom_logit_processor": (None,),
    }
    for name, allowed_values in unsupported.items():
        if sampling_params.get(name) not in allowed_values:
            raise ValueError(
                f"{name} is not supported with sampling-support replay because "
                "the trainer cannot reproduce its logit transformation"
            )
    return True


def configure_sampling_mask_request(request: dict[str, Any]) -> None:
    """Ask for a requested sampling mask as flat base64 arrays, unless the output store takes it."""
    inline = request.get("return_sampling_mask", False) and not request.get(OUTPUT_STORE_REQUEST_KEY, False)
    for key in ("return_flat_raw_sampling_mask", "return_flat_raw_sampling_mask_b64"):
        if inline:
            request[key] = True
        else:
            request.pop(key, None)


def decode_flat_sampling_mask(meta_info: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Support ids, per-token support lengths and log-probs from SGLang's base64 flat sampling mask."""

    def decode(key: str, dtype: type) -> np.ndarray:
        if f"{key}_b64" not in meta_info:
            raise ValueError(
                f"SGLang response is missing {key}_b64; use an SGLang build with return_flat_raw_sampling_mask, "
                "behind a router that forwards it"
            )
        # SGLang writes contiguous little-endian bytes; the dtype marker names only the width.
        wire_dtype = np.dtype(meta_info[f"{key}_b64_dtype"]).newbyteorder("<")
        raw = pybase64.b64decode(meta_info[f"{key}_b64"], validate=True)
        return np.frombuffer(raw, dtype=wire_dtype).astype(dtype)

    return (
        decode("output_token_sampling_mask_flat", np.int32),
        decode("output_token_sampling_mask_lengths", np.int32),
        decode("output_token_sampling_logprobs_flat", np.float32),
    )


def append_sampling_metadata(
    sample: Sample,
    output_token_ids: Sequence[int],
    meta_info: dict,
    *,
    aborted: bool = False,
    sampling_logprobs_mode: str = "selected",
) -> list[float]:
    """Append SGLang's realized support and return sampled-token log-probs."""
    finish_reason = meta_info.get("finish_reason") or {}
    no_output_abort = not output_token_ids and (aborted or finish_reason.get("type") == "abort")
    if no_output_abort and "output_token_sampling_mask_flat_b64" not in meta_info:
        _append_sampling_mask(sample, RolloutSamplingMask.from_mask_list([]))
        return []
    if sampling_logprobs_mode not in ("selected", "support"):
        raise ValueError(f"Unsupported sampling logprobs mode: {sampling_logprobs_mode}")
    support_mode = sampling_logprobs_mode == "support"

    ids, lengths, log_probs = decode_flat_sampling_mask(meta_info)
    tokens = np.asarray(output_token_ids, dtype=np.int32)
    if len(lengths) != len(tokens):
        raise ValueError(f"sampling support length {len(lengths)} != token length {len(tokens)}")
    if len(log_probs) != (len(ids) if support_mode else len(tokens)):
        raise ValueError(f"SGLang {sampling_logprobs_mode}-mode sampling log-probs do not align with the support")
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    validate_sampling_support(ids=ids, offsets=offsets, sampled_tokens=tokens, exactly_once=support_mode)
    _append_sampling_mask(sample, RolloutSamplingMask(ids=torch.from_numpy(ids), offsets=torch.from_numpy(offsets)))
    # One hit per row: support mode validated each sampled token as unique in its row.
    selected = log_probs[ids == np.repeat(tokens, lengths)] if support_mode else log_probs
    return selected.tolist()


def read_sampled_logprobs(
    meta_info: dict, output_token_ids: Sequence[int], *, sampling_logprobs_mode: str
) -> list[float]:
    """The sampled tokens' log-probs, one per token, when the sampling mask went to the output store."""
    if sampling_logprobs_mode == "support":
        # SGLang picked each sampled token's entry out of its support row.
        key = "output_token_sampling_selected_logprobs"
    elif sampling_logprobs_mode == "selected":
        key = "output_token_sampling_logprobs"
    else:
        raise ValueError(f"Unsupported sampling logprobs mode: {sampling_logprobs_mode}")
    log_probs = meta_info.get(key, [])
    if len(log_probs) != len(output_token_ids):
        raise StorePlanError(
            f"SGLang response {key} has {len(log_probs)} values for {len(output_token_ids)} output tokens"
        )
    return [float(value) for value in log_probs]


def append_forced_sampling_tokens(sample: Sample, token_ids: Sequence[int]) -> None:
    """Record singleton support for non-sampled tokens inserted by the environment."""
    sampling_mask = RolloutSamplingMask.from_mask_list([[int(token_id)] for token_id in token_ids])
    _append_sampling_mask(sample, sampling_mask)


def merge_sampling_masks(
    first: Sample,
    observation_token_ids: Sequence[int],
    second: Sample,
) -> RolloutSamplingMask | None:
    """Merge two per-response ragged masks with forced observation tokens between them."""
    first_mask = first.rollout_sampling_mask
    second_mask = second.rollout_sampling_mask
    if first_mask is None or second_mask is None:
        if first_mask is None and second_mask is None:
            return None
        raise ValueError("cannot merge samples unless both turns carry a complete rollout sampling mask")

    observation_mask = RolloutSamplingMask.from_mask_list([[int(token_id)] for token_id in observation_token_ids])
    return RolloutSamplingMask.concatenate((first_mask, observation_mask, second_mask))


def _append_sampling_mask(sample: Sample, sampling_mask: RolloutSamplingMask) -> None:
    if sample.rollout_sampling_mask is None:
        if sample.response_length != 0:
            raise ValueError("cannot initialize a sampling mask after response tokens have already been appended")
        sample.rollout_sampling_mask = sampling_mask
        return

    if len(sample.rollout_sampling_mask) != sample.response_length:
        raise ValueError(
            f"sampling mask length {len(sample.rollout_sampling_mask)} is not aligned with "
            f"response_length {sample.response_length} before appending"
        )
    sample.rollout_sampling_mask = RolloutSamplingMask.concatenate((sample.rollout_sampling_mask, sampling_mask))
