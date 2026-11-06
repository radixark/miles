"""The synthetic batch a fixed-shape (TPU) trainer compiles its first step on, before rollout 0 (--compile-warmup)."""

from typing import Any


def warmup_sample_lengths(
    *, bucket: int, pad_size: int, max_tokens_per_gpu: int, num_samples: int
) -> list[int] | None:
    """This rank's sample lengths for one optimizer step whose microbatch pads to `bucket`; None when none can.

    `get_batch` pads a packed microbatch to `pad_size`, so the samples share a budget rounded down to it. A budget
    above the packer's cap goes to one sample, which gets a microbatch of its own; the other samples are the shortest
    possible (a prompt and one response token) and pack into a microbatch of the smallest bucket.
    """
    budget = bucket - bucket % pad_size
    if budget > max_tokens_per_gpu:
        return [budget] + [2] * (num_samples - 1)
    length = budget // num_samples
    # shortcut: a bucket the samples would overshoot at two tokens each is left to compile at its first real step
    return [length] * num_samples if length >= 2 else None


def warmup_rollout_data(lengths: list[int]) -> dict[str, Any]:
    """This rank's shard, as the rollout side packages one: every sample is zero tokens with a one-token prompt."""
    loss_masks = [[1] * (length - 1) for length in lengths]
    return {
        "tokens": [[0] * length for length in lengths],
        "response_lengths": [length - 1 for length in lengths],
        "rewards": [0.0] * len(lengths),
        "raw_reward": [0.0] * len(lengths),
        "truncated": [0] * len(lengths),
        "sample_indices": list(range(len(lengths))),
        "rollout_ids": list(range(len(lengths))),
        "loss_masks": loss_masks,
        "rollout_mask_sums": [sum(mask) for mask in loss_masks],
        # sglang returns them with every sample, so the trainer's graphs take them as an input
        "rollout_log_probs": [[0.0] * (length - 1) for length in lengths],
        "total_lengths": list(lengths),
        "partition": list(range(len(lengths))),
    }
