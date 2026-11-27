"""The synthetic rollout shard a fixed-shape (TPU) trainer compiles its first step on (--compile-warmup)."""

from typing import Any

import numpy as np

# a one-token prompt and one response token
_MIN_SAMPLE_LENGTH = 2


def warmup_shard(num_samples: int, routed_experts_shape: tuple[int, int] | None = None) -> dict[str, Any]:
    """One optimizer step's worth of the shortest samples, keyed as the rollout side ships a shard: zero tokens,
    a one-token prompt, the rest response.

    Every fixed-shape microbatch pads to --seq-length, so any batch compiles the step's programs.
    `routed_experts_shape` is `(num_layers, top_k)` under rollout routing replay, whose steps compile the replay
    programs: each sample then carries the engine's routing rows.
    """
    lengths = [_MIN_SAMPLE_LENGTH] * num_samples
    response_lengths = [length - 1 for length in lengths]
    sample_ids = list(range(len(lengths)))
    zeros = [0.0] * len(lengths)
    shard = {
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
    if routed_experts_shape is not None:
        # replay's padding rows: the routers spread them across the experts, and the --ci-test
        # replay check skips them
        shard["rollout_routed_experts"] = [
            np.full((length - 1, *routed_experts_shape), -1, dtype=np.int32) for length in lengths
        ]
    return shard
