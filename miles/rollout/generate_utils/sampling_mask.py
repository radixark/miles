from argparse import Namespace
from collections.abc import Mapping, Sequence

from miles.utils.sampling import sampling_mask_replay_enabled, uses_sampling_support_truncation
from miles.utils.types import Sample


def _sampling_param_or_default(params: Mapping[str, object], name: str, default: object) -> object:
    value = params.get(name)
    return default if value is None else value


def should_return_sampling_mask(
    args: Namespace,
    sampling_params: Mapping[str, object] | None = None,
    *,
    evaluation: bool = False,
) -> bool:
    """Validate one request against the configured rollout-support contract."""
    if evaluation:
        return False

    configured = sampling_mask_replay_enabled(args)
    params = sampling_params or {}
    request_top_p = float(_sampling_param_or_default(params, "top_p", getattr(args, "rollout_top_p", 1.0)))
    request_top_k = int(_sampling_param_or_default(params, "top_k", getattr(args, "rollout_top_k", -1)))
    request_uses_truncation = uses_sampling_support_truncation(
        top_p=request_top_p,
        top_k=request_top_k,
        min_p=float(_sampling_param_or_default(params, "min_p", 0.0)),
    )
    if request_uses_truncation != configured:
        raise ValueError(
            "request-level top-p/top-k/min-p cannot change whether sampling is truncated; "
            "set --rollout-top-p/--rollout-top-k to enable rollout sampling-support replay for the run"
        )
    if configured:
        configured_top_k = int(getattr(args, "rollout_top_k", -1))
        if not 0 < request_top_k <= configured_top_k:
            raise ValueError(
                f"training request top_k must be in [1, {configured_top_k}] for rollout sampling-support replay"
            )
        configured_temperature = float(getattr(args, "rollout_temperature", 1.0))
        request_temperature = float(_sampling_param_or_default(params, "temperature", configured_temperature))
        if request_temperature != configured_temperature:
            raise ValueError(
                f"request temperature {request_temperature} does not match "
                f"--rollout-temperature {configured_temperature}"
            )

        unsupported = {
            "frequency_penalty": (0, 0.0, None),
            "presence_penalty": (0, 0.0, None),
            "repetition_penalty": (1, 1.0, None),
            "logit_bias": ({}, None),
            "custom_logit_processor": (None,),
        }
        for name, allowed_values in unsupported.items():
            if params.get(name) not in allowed_values:
                raise ValueError(
                    f"{name} is not supported with rollout sampling-support replay because "
                    "the trainer cannot reproduce its logit transformation"
                )
    return configured


def set_sampling_mask_request_defaults(
    sampling_params: dict[str, object],
    *,
    top_p: float,
    top_k: int,
    temperature: float,
) -> None:
    """Materialize the sampling values that actor replay assumes."""
    if sampling_params.get("top_p") is None:
        sampling_params["top_p"] = top_p
    if sampling_params.get("top_k") is None:
        sampling_params["top_k"] = top_k
    if sampling_params.get("temperature") is None:
        sampling_params["temperature"] = temperature


def _flatten_sampling_supports(
    token_ids: Sequence[int],
    supports: Sequence[Sequence[int]],
) -> tuple[list[int], list[int]]:
    """Flatten one sampling support per token into ids plus CSR-style offsets."""
    if len(token_ids) != len(supports):
        raise ValueError(f"sampling support length {len(supports)} != token length {len(token_ids)}")

    flat_ids: list[int] = []
    offsets = [0]
    for token_id, support in zip(token_ids, supports, strict=True):
        if not support:
            raise ValueError("sampling support must contain at least one token")
        if int(token_id) not in support:
            raise ValueError(f"sampled token {token_id} is absent from its sampling support")
        flat_ids.extend(support)
        offsets.append(len(flat_ids))
    return flat_ids, offsets


def append_sampling_metadata(
    sample: Sample,
    output_token_ids: Sequence[int],
    meta_info: dict,
) -> list[float]:
    """Append native SGLang support data and return its normalized log-probs."""
    supports = meta_info.get("output_token_sampling_mask")
    log_probs = meta_info.get("output_token_sampling_logprobs")
    if supports is None or log_probs is None:
        finish_reason = meta_info.get("finish_reason") or {}
        if finish_reason.get("type") == "abort" and not output_token_ids:
            return []
        raise ValueError(
            "SGLang response is missing output_token_sampling_mask or "
            "output_token_sampling_logprobs; use an SGLang build with the "
            "native return_sampling_mask primitive"
        )
    if len(log_probs) != len(output_token_ids):
        raise ValueError(f"sampling log-prob length {len(log_probs)} != output token length {len(output_token_ids)}")

    flat_ids, offsets = _flatten_sampling_supports(output_token_ids, supports)
    _append_flat_sampling_mask(sample, flat_ids, offsets)
    return [float(value) for value in log_probs]


def append_forced_sampling_tokens(sample: Sample, token_ids: Sequence[int]) -> None:
    """Record singleton support for non-sampled tokens inserted by the environment."""
    ids = [int(token_id) for token_id in token_ids]
    _append_flat_sampling_mask(sample, ids, list(range(len(ids) + 1)))


def merge_sampling_masks(
    first: Sample,
    observation_token_ids: Sequence[int],
    second: Sample,
) -> tuple[list[int] | None, list[int] | None]:
    """Merge two per-response ragged masks with forced observation tokens between them."""
    first_ids = first.rollout_sampling_mask_ids
    first_offsets = first.rollout_sampling_mask_offsets
    second_ids = second.rollout_sampling_mask_ids
    second_offsets = second.rollout_sampling_mask_offsets
    if first_ids is None or first_offsets is None or second_ids is None or second_offsets is None:
        if first_ids is None and first_offsets is None and second_ids is None and second_offsets is None:
            return None, None
        raise ValueError("cannot merge samples unless both turns carry a complete rollout sampling mask")

    observation_ids = [int(token_id) for token_id in observation_token_ids]
    merged_ids = [
        *first_ids,
        *observation_ids,
        *second_ids,
    ]
    first_end = len(first_ids)
    observation_end = first_end + len(observation_ids)
    merged_offsets = [
        *first_offsets,
        *(first_end + offset for offset in range(1, len(observation_ids) + 1)),
        *(observation_end + offset for offset in second_offsets[1:]),
    ]
    return merged_ids, merged_offsets


def _append_flat_sampling_mask(sample: Sample, flat_ids: list[int], offsets: list[int]) -> None:
    if sample.rollout_sampling_mask_offsets is None:
        if sample.rollout_sampling_mask_ids is not None:
            raise ValueError("rollout_sampling_mask_ids is set without offsets")
        if sample.response_length != 0:
            raise ValueError("cannot initialize a sampling mask after response tokens have already been appended")
        sample.rollout_sampling_mask_ids = []
        sample.rollout_sampling_mask_offsets = [0]

    if sample.rollout_sampling_mask_ids is None:
        raise ValueError("rollout_sampling_mask_offsets is set without ids")
    if len(sample.rollout_sampling_mask_offsets) != sample.response_length + 1:
        raise ValueError(
            "sampling mask offsets must be aligned before appending: "
            f"got {len(sample.rollout_sampling_mask_offsets)} offsets for "
            f"response_length={sample.response_length}"
        )

    base = len(sample.rollout_sampling_mask_ids)
    sample.rollout_sampling_mask_ids.extend(flat_ids)
    sample.rollout_sampling_mask_offsets.extend(base + offset for offset in offsets[1:])
