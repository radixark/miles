from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.rollout.generate_utils.sampling_mask import (
    append_forced_sampling_tokens,
    append_sampling_metadata,
    merge_sampling_masks,
)
from miles.utils.types import Sample


@pytest.mark.parametrize(
    ("rollout_top_p", "request_top_p", "expected"),
    [
        (0.95, 0.95, True),
        (1.0, 1.0, False),
        (0.95, 1.0, False),
    ],
)
def test_generate_payload_automatically_requests_top_p_sampling_mask(rollout_top_p, request_top_p, expected):
    args = SimpleNamespace(
        rollout_top_p=rollout_top_p,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
    )

    payload, halt_status = compute_request_payload(
        args,
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 4, "top_p": request_top_p},
    )

    assert halt_status is None
    assert payload.get("return_sampling_mask", False) is expected


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
