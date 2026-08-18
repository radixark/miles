"""Compute Megatron/FSDP scheduler `train_iters` from rollout settings."""

from argparse import Namespace


def compute_train_iters(args: Namespace) -> int:
    """Return the iteration count used to size the LR/WD schedule.

    Eval-only (`--num-rollout 0`) still constructs the scheduler, and Megatron
    asserts `lr_decay_steps > 0`, so that case is 1 rather than 0.
    """
    if args.num_rollout == 0:
        return 1
    estimated = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if estimated <= 0:
        total_samples = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt
        raise ValueError(
            f"Invalid training configuration: total samples ({total_samples}) "
            f"must be at least global_batch_size ({args.global_batch_size})."
        )
    return estimated
