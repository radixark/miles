from argparse import Namespace
from collections.abc import Mapping, Sequence

from miles.utils.types import Sample


def should_return_sampling_mask(
    args: Namespace,
    sampling_params: Mapping[str, object] | None = None,
) -> bool:
    """Whether this request needs exact truncated-sampling normalization."""
    rollout_top_p = args.rollout_top_p
    request_top_p = float((sampling_params or {}).get("top_p", rollout_top_p))
    return rollout_top_p < 1.0 and request_top_p < 1.0


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
        support_ids = [int(value) for value in support]
        if not support_ids:
            raise ValueError("sampling support must contain at least one token")
        if int(token_id) not in support_ids:
            raise ValueError(f"sampled token {token_id} is absent from its sampling support")
        flat_ids.extend(support_ids)
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
        raise ValueError(
            "SGLang response is missing output_token_sampling_mask or output_token_sampling_logprobs; use an SGLang build with the native return_sampling_mask primitive"
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
    observation_offsets = list(range(len(observation_ids) + 1))
    merged_ids = [
        *first_ids,
        *observation_ids,
        *second_ids,
    ]
    first_end = len(first_ids)
    observation_end = first_end + len(observation_ids)
    merged_offsets = [
        *first_offsets,
        *(first_end + offset for offset in observation_offsets[1:]),
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
            f"sampling mask offsets must be aligned before appending: got {len(sample.rollout_sampling_mask_offsets)} offsets for response_length={sample.response_length}"
        )

    base = len(sample.rollout_sampling_mask_ids)
    sample.rollout_sampling_mask_ids.extend(flat_ids)
    sample.rollout_sampling_mask_offsets.extend(base + offset for offset in offsets[1:])
