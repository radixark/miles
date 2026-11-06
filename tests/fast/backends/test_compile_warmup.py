import pytest

from miles.backends.torchtitan_utils.compile_warmup import warmup_rollout_data, warmup_sample_lengths
from miles.backends.torchtitan_utils.length_buckets import fixed_shape_length_bucket, fixed_shape_length_buckets
from miles.backends.training_utils.data.rollout import _num_microbatches_within_cap
from miles.utils.data import padded_token_count


def _microbatch_buckets(lengths: list[int], *, max_tokens_per_gpu: int, pad_size: int, seq_length: int) -> list[int]:
    """The bucket of each microbatch the packer makes of this rank's samples (first fit, padded as get_batch pads)."""
    microbatches: list[list[int]] = []
    for length in lengths:
        for microbatch in microbatches:
            if sum(microbatch) + length <= max_tokens_per_gpu:
                microbatch.append(length)
                break
        else:
            microbatches.append([length])
    assert len(microbatches) == _num_microbatches_within_cap(lengths, max_tokens_per_gpu)
    return sorted(fixed_shape_length_bucket(padded_token_count(sum(m), pad_size), seq_length) for m in microbatches)


@pytest.mark.parametrize(
    ("seq_length", "max_tokens_per_gpu", "num_samples", "pad_size", "unwarmed"),
    [
        # the TPU launcher: the cap is the sequence length, 128 samples over 4 ranks
        (8192, 8192, 32, 128, set()),
        # a cap below --seq-length (field notes: 7680 vs 8192): the top bucket gets a lone sample
        (8192, 7680, 32, 128, set()),
        # a --seq-length off the power-of-two ladder, and off the pad, is the top bucket itself
        (3000, 3000, 3, 128, set()),
        (5000, 4500, 4, 256, set()),
        # 1024 local samples of two tokens each already overshoot the smallest bucket
        (4096, 4096, 1024, 128, {1024}),
    ],
)
def test_each_bucket_gets_a_microbatch_of_its_own(seq_length, max_tokens_per_gpu, num_samples, pad_size, unwarmed):
    """The trainer then compiles exactly the programs of each bucket, for one optimizer step's worth of samples."""
    warmed = set()
    for bucket in fixed_shape_length_buckets(seq_length):
        lengths = warmup_sample_lengths(
            bucket=bucket, pad_size=pad_size, max_tokens_per_gpu=max_tokens_per_gpu, num_samples=num_samples
        )
        if lengths is None:
            continue
        assert len(lengths) == num_samples and min(lengths) >= 2
        buckets = _microbatch_buckets(
            lengths, max_tokens_per_gpu=max_tokens_per_gpu, pad_size=pad_size, seq_length=seq_length
        )
        assert bucket in buckets and len(buckets) <= 2, (bucket, buckets)
        warmed.add(bucket)
    assert warmed == fixed_shape_length_buckets(seq_length) - unwarmed


def test_the_shard_looks_like_a_rollout_of_zero_tokens():
    shard = warmup_rollout_data([4, 3])

    assert shard["tokens"] == [[0, 0, 0, 0], [0, 0, 0]]
    assert shard["response_lengths"] == [3, 2]
    assert shard["loss_masks"] == [[1, 1, 1], [1, 1]]
    assert shard["rollout_mask_sums"] == [3, 2]
    assert shard["rollout_log_probs"] == [[0.0] * 3, [0.0] * 2]
    assert shard["total_lengths"] == [4, 3] and shard["partition"] == [0, 1]
    assert shard["rewards"] == [0.0, 0.0] and shard["sample_indices"] == [0, 1]
