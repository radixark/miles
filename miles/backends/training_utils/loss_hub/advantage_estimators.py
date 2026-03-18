from argparse import Namespace

import torch

from miles.utils.ppo_utils import (
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)

from ..parallel import ParallelState


def compute_advantages(
    args: Namespace,
    parallel_state: ParallelState,
    kl: list[torch.Tensor],
    rewards: list[float],
    log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    values: list[torch.Tensor] | None = None,
    teacher_log_probs: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Dispatch to the configured advantage estimator.

    Returns:
        (advantages, returns) — both lists of tensors, one per sample.
    """
    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = parallel_state.cp_rank
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd, parallel_state
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
            parallel_state=parallel_state,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    elif args.advantage_estimator == "on_policy_distillation":
        assert teacher_log_probs is not None, "teacher_log_probs required for on_policy_distillation"
        device = log_probs[0].device
        teacher_log_probs = [t_log_prob.to(device=device) for t_log_prob in teacher_log_probs]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        advantages = [
            teacher_log_prob - student_log_prob
            for teacher_log_prob, student_log_prob in zip(teacher_log_probs, log_probs, strict=False)
        ]
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    return advantages, returns
