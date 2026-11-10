"""The synthetic rollout shards a fixed-shape (TPU) trainer compiles its first step on (--compile-warmup)."""

from collections.abc import Iterator
from typing import Any

from miles.utils.data import fixed_shape_length_buckets

# a one-token prompt and one response token
_MIN_SAMPLE_LENGTH = 2


def warmup_shards(
    *, seq_len: int, pad_size: int, max_tokens_per_gpu: int, num_samples: int
) -> Iterator[tuple[int, dict[str, Any]]]:
    """One optimizer step's worth of samples per length bucket, sized so the packer gives the bucket a microbatch.

    A bucket the samples overshoot even at their minimum length is skipped; it compiles at its first real step.
    """
    for bucket in sorted(fixed_shape_length_buckets(seq_len)):
        lengths = _sample_lengths(
            bucket, pad_size=pad_size, max_tokens_per_gpu=max_tokens_per_gpu, num_samples=num_samples
        )
        if lengths is not None:
            yield bucket, _zero_token_shard(lengths)


def _sample_lengths(bucket: int, *, pad_size: int, max_tokens_per_gpu: int, num_samples: int) -> list[int] | None:
    # get_batch pads a packed microbatch up to pad_size, so its tokens must round down to it to stay in the bucket
    tokens = bucket - bucket % pad_size
    if tokens > max_tokens_per_gpu:
        # the packer gives a sample above its cap a microbatch of its own; the rest share the smallest bucket
        return [tokens] + [_MIN_SAMPLE_LENGTH] * (num_samples - 1)
    if tokens < _MIN_SAMPLE_LENGTH * num_samples:
        return None
    return [tokens // num_samples] * num_samples


def _zero_token_shard(lengths: list[int]) -> dict[str, Any]:
    """A shard keyed as the rollout side ships one: zero tokens, a one-token prompt, the rest response."""
    response_lengths = [length - 1 for length in lengths]
    sample_ids = list(range(len(lengths)))
    zeros = [0.0] * len(lengths)
    return {
        "tokens": [[0] * length for length in lengths],
        "total_lengths": list(lengths),
        "response_lengths": response_lengths,
        "loss_masks": [[1] * n for n in response_lengths],
        "rollout_mask_sums": list(response_lengths),
        # sglang returns them with every sample, so the trainer's graphs take them as an input
        "rollout_log_probs": [[0.0] * n for n in response_lengths],
        "rewards": zeros,
        "raw_reward": list(zeros),
        "truncated": [0] * len(lengths),
        "sample_indices": sample_ids,
        "rollout_ids": list(sample_ids),
        "partition": list(sample_ids),
    }
