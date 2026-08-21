from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.sampling_mask import (
    append_forced_sampling_tokens,
    append_sampling_metadata,
    merge_sampling_masks,
)
from miles.utils.sampling import sampling_mask_replay_enabled
from miles.utils.types import Sample


@pytest.mark.parametrize(
    ("top_p", "top_k", "expected"),
    [(0.95, -1, True), (1.0, 32, True), (1.0, -1, False)],
)
def test_sampling_mask_replay_enabled_for_any_truncated_support(top_p, top_k, expected):
    args = SimpleNamespace(rollout_top_p=top_p, rollout_top_k=top_k)
    assert sampling_mask_replay_enabled(args) is expected


def test_append_sampling_metadata_preserves_ragged_support_and_native_logprobs():
    sample = Sample(tokens=[1, 2])
    meta_info = {
        "output_token_sampling_mask": [[10, 4, 7], [11, 3]],
        "output_token_sampling_logprobs": [-0.25, -0.5],
    }

    log_probs = append_sampling_metadata(sample, [10, 11], meta_info)
    sample.tokens.extend([10, 11])
    sample.response_length = 2

    assert log_probs == [-0.25, -0.5]
    assert sample.rollout_sampling_mask_ids == [10, 4, 7, 11, 3]
    assert sample.rollout_sampling_mask_offsets == [0, 3, 5]
    sample.validate()


def test_forced_tokens_append_singleton_support_and_strip_cleanly():
    sample = Sample(
        tokens=[1, 10],
        response_length=1,
        rollout_sampling_mask_ids=[10, 4],
        rollout_sampling_mask_offsets=[0, 2],
    )

    append_forced_sampling_tokens(sample, [20, 21])
    sample.tokens.extend([20, 21])
    sample.response_length += 2
    sample.validate()

    assert sample.rollout_sampling_mask_ids == [10, 4, 20, 21]
    assert sample.rollout_sampling_mask_offsets == [0, 2, 3, 4]

    tokenizer = type("Tokenizer", (), {"decode": staticmethod(lambda _: "")})()
    sample.strip_last_output_tokens(2, tokenizer)
    assert sample.rollout_sampling_mask_ids == [10, 4]
    assert sample.rollout_sampling_mask_offsets == [0, 2]


def test_merge_sampling_masks_inserts_singleton_observation_supports():
    first = Sample(
        response_length=1,
        rollout_sampling_mask_ids=[10, 4],
        rollout_sampling_mask_offsets=[0, 2],
    )
    second = Sample(
        response_length=1,
        rollout_sampling_mask_ids=[30, 7, 8],
        rollout_sampling_mask_offsets=[0, 3],
    )

    ids, offsets = merge_sampling_masks(first, [20, 21], second)

    assert ids == [10, 4, 20, 21, 30, 7, 8]
    assert offsets == [0, 2, 3, 4, 7]


def test_append_sampling_metadata_rejects_support_without_sampled_token():
    with pytest.raises(ValueError, match="sampled token 10 is absent"):
        append_sampling_metadata(
            Sample(),
            [10],
            {
                "output_token_sampling_mask": [[4, 7]],
                "output_token_sampling_logprobs": [-0.25],
            },
        )


def test_abort_before_sampling_does_not_require_sampling_metadata():
    assert (
        append_sampling_metadata(
            Sample(),
            [],
            {"finish_reason": {"type": "abort"}},
        )
        == []
    )
