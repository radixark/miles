from collections.abc import Sequence

from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


def append_forced_sampling_tokens(sample: Sample, token_ids: Sequence[int]) -> None:
    """Record singleton support for non-sampled tokens inserted by the environment."""
    sampling_mask = RolloutSamplingMask.from_mask_list([[int(token_id)] for token_id in token_ids])
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
