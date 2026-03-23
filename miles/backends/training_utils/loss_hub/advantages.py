from argparse import Namespace

import torch

from miles.utils.distributed_utils import distributed_masked_whiten
from miles.utils.ppo_utils import (
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)

from ..cp_utils import get_logits_and_tokens_offset_with_cp
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


def normalize_advantages(
    args,
    parallel_state,
    advantages,
    loss_masks,
    total_lengths,
    response_lengths,
    max_seq_lens=None,
):
    all_advs = torch.cat(advantages)
    cp_size = parallel_state.cp_size
    if cp_size == 1:
        all_masks = torch.cat(loss_masks)
    else:
        mask_chunks = []
        for i in range(len(advantages)):
            total_len = total_lengths[i]
            response_len = response_lengths[i]
            prompt_len = total_len - response_len
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

            _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                total_len, response_len, parallel_state, args.qkv_format, max_seq_len
            )

            # Convert global offsets to response-space offsets
            s0, e0 = token_offsets[0]
            s1, e1 = token_offsets[1]
            res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
            res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

            local_mask_parts = []
            full_mask = loss_masks[i]
            if res_e0 > res_s0:
                local_mask_parts.append(full_mask[res_s0:res_e0])
            if res_e1 > res_s1:
                local_mask_parts.append(full_mask[res_s1:res_e1])

            # Concatenate the parts to form the final mask chunk for this rank and this sequence
            local_mask_chunk = (
                torch.cat(local_mask_parts)
                if local_mask_parts
                else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
            )
            mask_chunks.append(local_mask_chunk)

        all_masks = torch.cat(mask_chunks)

    if all_masks.numel() > 0:
        assert (
            all_advs.size() == all_masks.size()
        ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
        dp_group = parallel_state.dp_group

        whitened_advs_flat = distributed_masked_whiten(
            all_advs,
            all_masks,
            process_group=dp_group,
            shift_mean=True,
        )
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    return advantages
