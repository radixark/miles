"""Shared prompt and trajectory grouping for reward processing."""

from argparse import Namespace

from miles.utils.types import Sample


def reward_group_segments(
    args: Namespace, samples: list[Sample], prompt_group_sizes: list[int] | None = None
) -> list[list[int]]:
    """Return flattened row indices for each prompt, including interleaved batches."""
    if prompt_group_sizes is not None:
        assert sum(prompt_group_sizes) == len(
            samples
        ), f"prompt group sizes sum to {sum(prompt_group_sizes)}, but got {len(samples)} rewards"
        groups: list[list[int]] = []
        start = 0
        for size in prompt_group_sizes:
            if size > 0:
                groups.append(list(range(start, start + size)))
            start += size
        return groups
    if all(sample.group_index is not None for sample in samples):
        by_prompt: dict[int, list[int]] = {}
        for row, sample in enumerate(samples):
            by_prompt.setdefault(int(sample.group_index), []).append(row)
        return list(by_prompt.values())
    if len(samples) == args.n_samples_per_prompt * args.rollout_batch_size:
        return [
            list(range(start, start + args.n_samples_per_prompt))
            for start in range(0, len(samples), args.n_samples_per_prompt)
        ]
    return [list(range(len(samples)))]


def rollout_segments(samples: list[Sample], rows: list[int]) -> list[list[int]]:
    """Collect sibling training rows belonging to one generated trajectory."""
    by_rollout: dict[int | tuple[str, int], list[int]] = {}
    for row in rows:
        sample = samples[row]
        key = sample.rollout_id if sample.rollout_id is not None else sample.index
        by_rollout.setdefault(key if key is not None else ("row", row), []).append(row)
    return list(by_rollout.values())
