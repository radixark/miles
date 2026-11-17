"""The synthetic rollout shard a fixed-shape (TPU) trainer compiles its first step on (--compile-warmup)."""

from typing import Any

# a one-token prompt and one response token
_MIN_SAMPLE_LENGTH = 2


def warmup_shard(num_samples: int) -> dict[str, Any]:
    """One optimizer step's worth of the shortest samples, keyed as the rollout side ships a shard: zero tokens,
    a one-token prompt, the rest response.

    Every fixed-shape microbatch pads to --seq-length, so any batch compiles the step's programs.
    """
    lengths = [_MIN_SAMPLE_LENGTH] * num_samples
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
