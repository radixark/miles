import math
from typing import Any

import ray
import torch

from miles.utils.iter_utils import group_by
from miles.utils.ray_utils import Box
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.types import Sample


def convert_samples_to_train_data(
    args,
    samples: list[Sample] | list[list[Sample]],
    metadata: dict[str, Any],
    custom_convert_samples_to_train_data_func,
    custom_reward_post_process_func,
):
    """
    Convert inference generated samples to training data.
    """
    if (f := custom_convert_samples_to_train_data_func) is not None:
        return f(args, samples)

    raw_rewards, rewards = _post_process_rewards(
        args, samples, custom_reward_post_process_func=custom_reward_post_process_func
    )

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    if args.loss_aggregation == "prompt_mean":
        prompt_group_indices = [sample.group_index for sample in samples]
        if None in prompt_group_indices:
            raise ValueError("--loss-aggregation prompt_mean requires every Sample.group_index to be set.")
        prompt_groups = group_by(zip(prompt_group_indices, loss_masks, strict=True), key=lambda pair: pair[0])
        partial_groups = {
            group_index: len(group)
            for group_index, group in prompt_groups.items()
            if len(group) != args.n_samples_per_prompt
        }
        if partial_groups:
            raise ValueError(
                "--loss-aggregation prompt_mean requires complete prompt groups; "
                f"expected {args.n_samples_per_prompt} samples per group, got {partial_groups}."
            )
        group_mask_totals = {
            group_index: sum(sum(mask) for _, mask in group) for group_index, group in prompt_groups.items()
        }
        train_data["prompt_group_indices"] = prompt_group_indices
        train_data["prompt_mask_sums"] = [group_mask_totals[group_index] for group_index in prompt_group_indices]

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

    if samples[0].rollout_indexer_topk is not None:
        train_data["rollout_indexer_topk"] = [sample.rollout_indexer_topk for sample in samples]

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if any(sample.weight_versions for sample in samples):
        train_data["weight_versions"] = [sample.weight_versions for sample in samples]

    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    if samples[0].opd_reverse_kl is not None:
        train_data["opd_reverse_kl"] = [sample.opd_reverse_kl for sample in samples]

    x = metadata.get("dynamic_global_batch_size")
    assert args.use_dynamic_global_batch_size == (x is not None)
    if x is not None:
        train_data["dynamic_global_batch_size"] = x

    return train_data


def _post_process_rewards(args, samples: list[Sample] | list[list[Sample]], custom_reward_post_process_func):
    if (f := custom_reward_post_process_func) is not None:
        return f(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        # group norm
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            # when samples count are not equal in each group
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, raw_rewards


def _prompt_group_partitions(
    prompt_group_indices: list[int],
    prompt_mask_sums: list[float],
    loss_masks: list[list[float]],
    total_lengths: list[int],
    dp_size: int,
    *,
    balance_data: bool,
    expected_group_size: int,
) -> list[list[int]]:
    num_samples = len(total_lengths)
    if len(prompt_group_indices) != num_samples:
        raise ValueError(
            "--loss-aggregation prompt_mean requires one prompt_group_indices entry per sample "
            f"(got {len(prompt_group_indices)} for {num_samples} samples)."
        )
    if len(prompt_mask_sums) != num_samples:
        raise ValueError(
            "--loss-aggregation prompt_mean requires one prompt_mask_sums entry per sample "
            f"(got {len(prompt_mask_sums)} for {num_samples} samples)."
        )
    if len(loss_masks) != num_samples:
        raise ValueError(f"--loss-aggregation prompt_mean got {len(loss_masks)} loss masks for {num_samples} samples.")

    group_to_indices = group_by(range(len(prompt_group_indices)), key=lambda i: prompt_group_indices[i])
    partial_groups = {
        group_index: len(indices)
        for group_index, indices in group_to_indices.items()
        if len(indices) != expected_group_size
    }
    if partial_groups:
        raise ValueError(
            "--loss-aggregation prompt_mean requires complete prompt groups; "
            f"expected {expected_group_size} samples per group, got {partial_groups}."
        )
    for group_index, indices in group_to_indices.items():
        expected_mask_sum = sum(sum(loss_masks[index]) for index in indices)
        if any(not math.isclose(float(prompt_mask_sums[index]), expected_mask_sum) for index in indices):
            raise ValueError(
                "--loss-aggregation prompt_mean received inconsistent prompt_mask_sums for group "
                f"{group_index!r}; expected {expected_mask_sum}."
            )
    group_keys = list(group_to_indices)

    if len(group_keys) % dp_size != 0:
        raise ValueError(
            "--loss-aggregation prompt_mean requires the number of prompt groups in a train step "
            f"to be divisible by dp_size (got {len(group_keys)} prompt groups, dp_size={dp_size})."
        )

    if balance_data:
        group_lengths = [sum(total_lengths[index] for index in indices) for indices in group_to_indices.values()]
        group_partitions = get_seqlen_balanced_partitions(group_lengths, dp_size, equal_size=True)
    else:
        group_partitions = [range(i, len(group_keys), dp_size) for i in range(dp_size)]

    return [
        [sample_index for group_pos in partition for sample_index in group_to_indices[group_keys[group_pos]]]
        for partition in group_partitions
    ]


def split_train_data_by_dp(args, data, dp_size):
    """Split the train data by data parallel size."""
    rollout_data_list = split_train_data_by_dp_raw(args, data, dp_size=dp_size)
    return [Box(ray.put(rollout_data)) for rollout_data in rollout_data_list]


def split_train_data_by_dp_raw(args, data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    """Split the train data by data parallel size."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths
    loss_aggregation = args.loss_aggregation

    if loss_aggregation == "token_mean":
        global_batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
        if len(data["loss_masks"]) % global_batch_size != 0:
            raise ValueError(
                f"Cannot split {len(data['loss_masks'])} loss masks into token_mean steps of {global_batch_size}."
            )

    if loss_aggregation == "prompt_mean":
        missing = {"prompt_group_indices", "prompt_mask_sums"} - data.keys()
        if missing:
            raise ValueError(
                "--loss-aggregation prompt_mean requires custom train data to include "
                f"{', '.join(sorted(missing))}."
            )
        # Each rank consumes contiguous global_batch_size // dp_size slices per
        # optimizer step, so with multiple steps per rollout a prompt group stays
        # within one step only if that per-rank slice is a multiple of the group
        # size. Dynamic global batch size always trains one step per rollout.
        if not args.use_dynamic_global_batch_size and len(total_lengths) > args.global_batch_size:
            num_local_gbs = args.global_batch_size // dp_size
            if num_local_gbs % args.n_samples_per_prompt != 0:
                raise ValueError(
                    "--loss-aggregation prompt_mean requires global_batch_size // dp_size "
                    f"({args.global_batch_size} // {dp_size} = {num_local_gbs}) to be a multiple of "
                    f"n_samples_per_prompt ({args.n_samples_per_prompt}) when a rollout spans multiple "
                    "train steps; otherwise a prompt group would straddle an optimizer-step boundary."
                )
        partitions = _prompt_group_partitions(
            data["prompt_group_indices"],
            data["prompt_mask_sums"],
            data["loss_masks"],
            total_lengths,
            dp_size,
            balance_data=args.balance_data,
            expected_group_size=args.n_samples_per_prompt,
        )
    elif args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    if loss_aggregation == "token_mean":
        if global_batch_size % dp_size != 0:
            raise ValueError(
                f"token_mean requires global_batch_size {global_batch_size} to be divisible by dp_size {dp_size}."
            )
        local_samples_per_step = global_batch_size // dp_size
        num_steps = len(data["loss_masks"]) // global_batch_size
        empty_steps = []
        for step in range(num_steps):
            step_masks = [
                data["loss_masks"][sample_index]
                for partition in partitions
                for sample_index in partition[step * local_samples_per_step : (step + 1) * local_samples_per_step]
            ]
            if sum(sum(mask) for mask in step_masks) <= 0:
                empty_steps.append(step)
        if empty_steps:
            raise ValueError(
                f"token_mean requires at least one active token per optimizer step; empty: {empty_steps}."
            )

    ans = []

    for i in range(dp_size):
        rollout_data = {}
        partition = partitions[i]
        rollout_data["partition"] = partition
        for key in [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "truncated",
            "loss_masks",
            "round_number",
            "sample_indices",
            "rollout_log_probs",
            "rollout_routed_experts",
            "rollout_indexer_topk",
            "prompt",
            "teacher_log_probs",
            "opd_reverse_kl",
            "seq_witness_ids",
            "weight_versions",
            "prompt_group_indices",
            "prompt_mask_sums",
        ]:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
            "dynamic_global_batch_size",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        ans.append(rollout_data)
    return ans
