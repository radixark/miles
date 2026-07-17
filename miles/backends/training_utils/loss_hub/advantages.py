from argparse import Namespace

import torch

from miles.backends.training_utils.cp_utils import get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.loss_hub.math_utils import (
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import distributed_masked_whiten


def compute_advantages(
    args: Namespace,
    kl: list[torch.Tensor],
    rewards: list[float],
    log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    values: list[torch.Tensor] | None = None,
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
        cp_rank = get_parallel_state().cp.rank
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
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

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    return advantages, returns


def normalize_advantages(
    args: Namespace,
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> list[torch.Tensor]:
    """Whiten advantages across the DP group using `loss_masks` for weighting.

    Under CP > 1 the mask is sliced to this rank's tokens; when the local
    mask is empty the inputs pass through unchanged. Output shapes match
    `advantages`.
    """
    num_samples = len(advantages)
    assert len(loss_masks) == num_samples
    assert len(total_lengths) == num_samples
    assert len(response_lengths) == num_samples
    if max_seq_lens is not None:
        assert len(max_seq_lens) == num_samples

    parallel_state = get_parallel_state()
    all_advs = torch.cat(advantages)
    cp_size = parallel_state.cp.size
    if cp_size == 1:
        all_masks = torch.cat(loss_masks)
    else:
        mask_chunks = []
        max_seq_lens_iter = max_seq_lens if max_seq_lens is not None else [None] * num_samples
        for total_len, response_len, full_mask, max_seq_len in zip(
            total_lengths, response_lengths, loss_masks, max_seq_lens_iter, strict=True
        ):
            prompt_len = total_len - response_len

            _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                total_len, response_len, args.qkv_format, max_seq_len
            )

            # Convert global offsets to response-space offsets
            (s0, e0), (s1, e1) = token_offsets
            res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
            res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

            local_mask_parts = []
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
        dp_group = parallel_state.effective_dp.group

        whitened_advs_flat = distributed_masked_whiten(
            all_advs,
            all_masks,
            process_group=dp_group,
            shift_mean=True,
        )
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    return advantages
