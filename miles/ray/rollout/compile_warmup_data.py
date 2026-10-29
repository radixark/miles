"""Synthetic rollouts that make the trainer compile its per-rollout passes before rollout 0 arrives (--compile-warmup)."""

from miles.backends.torchtitan_utils.length_buckets import fixed_shape_length_bucket, fixed_shape_length_buckets
from miles.utils.types import Sample


def compile_warmup_samples(args, *, dp_size: int) -> dict[int, list[Sample]]:
    """One optimizer step's worth of samples per length bucket, each DP rank's share packing into one microbatch.

    Every sample has the same length, so any DP split gives each rank the same token sum, which lies in the
    bucket's range and under the packer's --max-tokens-per-gpu cap.
    """
    seq_length = args.raw_torchtitan.seq_length
    num_local_samples = args.global_batch_size // dp_size
    samples_of_bucket = {}
    for bucket in sorted(fixed_shape_length_buckets(seq_length)):
        length = min(bucket, args.max_tokens_per_gpu) // num_local_samples
        # One microbatch cannot reach a bucket above the cap (only a lone sample longer than the cap does), nor a
        # bucket the local samples overshoot with their minimum of one prompt and one response token each.
        if length >= 2 and fixed_shape_length_bucket(length * num_local_samples, seq_length) == bucket:
            samples_of_bucket[bucket] = _synthetic_samples(args, length=length)
    return samples_of_bucket


def _synthetic_samples(args, *, length: int) -> list[Sample]:
    response_length = length - 1
    reward = {args.reward_key: 0.0} if args.reward_key else 0.0
    return [
        Sample(
            group_index=index // args.n_samples_per_prompt,
            index=index,
            tokens=[0] * length,
            response_length=response_length,
            reward=reward,
            loss_mask=[1] * response_length,
            # sglang returns them with every sample, so the trainer's graphs take them as an input
            rollout_log_probs=[0.0] * response_length,
            status=Sample.Status.COMPLETED,
        )
        for index in range(args.global_batch_size)
    ]
