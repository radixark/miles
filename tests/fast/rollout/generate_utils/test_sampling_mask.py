from miles.rollout.generate_utils.sampling_mask import append_forced_sampling_tokens, merge_sampling_masks
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample


def test_forced_tokens_append_singleton_support_and_strip_cleanly():
    sample = Sample(
        tokens=[1, 10],
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[10, 4], offsets=[0, 2]),
    )

    append_forced_sampling_tokens(sample, [20, 21])
    sample.tokens.extend([20, 21])
    sample.response_length += 2
    sample.validate()

    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4, 20, 21]
    assert offsets.tolist() == [0, 2, 3, 4]

    tokenizer = type("Tokenizer", (), {"decode": staticmethod(lambda _: "")})()
    sample.strip_last_output_tokens(2, tokenizer)
    ids, offsets = sample.rollout_sampling_mask._as_tensors()
    assert ids.tolist() == [10, 4]
    assert offsets.tolist() == [0, 2]


def test_merge_sampling_masks_inserts_singleton_observation_supports():
    first = Sample(
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[10, 4], offsets=[0, 2]),
    )
    second = Sample(
        response_length=1,
        rollout_sampling_mask=RolloutSamplingMask(ids=[30, 7, 8], offsets=[0, 3]),
    )

    sampling_mask = merge_sampling_masks(first, [20, 21], second)
    ids, offsets = sampling_mask._as_tensors()

    assert ids.tolist() == [10, 4, 20, 21, 30, 7, 8]
    assert offsets.tolist() == [0, 2, 3, 4, 7]
