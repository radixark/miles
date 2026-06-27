from typing import Any

import ray
import torch

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

    if getattr(args, "loss_aggregation", "sample_mean") == "prompt_mean":
        group_mask_totals: dict[int, int] = {}
        prompt_group_indices: list[int] = []
        for sample, loss_mask in zip(samples, loss_masks, strict=True):
            if sample.group_index is None:
                raise ValueError("--loss-aggregation prompt_mean requires every Sample.group_index to be set.")
            prompt_group_indices.append(sample.group_index)
            group_mask_totals[sample.group_index] = group_mask_totals.get(sample.group_index, 0) + sum(loss_mask)
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
    total_lengths: list[int],
    dp_size: int,
    *,
    balance_data: bool,
) -> list[list[int]]:
    if len(prompt_group_indices) != len(total_lengths):
        raise ValueError(
            "--loss-aggregation prompt_mean requires one prompt_group_indices entry per sample "
            f"(got {len(prompt_group_indices)} for {len(total_lengths)} samples)."
        )

    group_to_indices: dict[int, list[int]] = {}
    group_order: list[int] = []
    for sample_index, group_index in enumerate(prompt_group_indices):
        group_key = int(group_index.item()) if isinstance(group_index, torch.Tensor) else int(group_index)
        if group_key not in group_to_indices:
            group_to_indices[group_key] = []
            group_order.append(group_key)
        group_to_indices[group_key].append(sample_index)

    if len(group_order) % dp_size != 0:
        raise ValueError(
            "--loss-aggregation prompt_mean requires the number of prompt groups in a train step "
            f"to be divisible by dp_size (got {len(group_order)} prompt groups, dp_size={dp_size})."
        )

    if balance_data:
        group_lengths = [
            sum(total_lengths[index] for index in group_to_indices[group_key]) for group_key in group_order
        ]
        group_partitions = get_seqlen_balanced_partitions(group_lengths, dp_size, equal_size=True)
    else:
        group_partitions = [range(i, len(group_order), dp_size) for i in range(dp_size)]

    return [
        [sample_index for group_index in partition for sample_index in group_to_indices[group_order[group_index]]]
        for partition in group_partitions
    ]


def split_train_data_by_dp(args, data, dp_size):
    """Split the train data by data parallel size."""
    rollout_data = {}

    if "prompt" in data:
        rollout_data["prompt"] = data["prompt"]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if getattr(args, "loss_aggregation", "sample_mean") == "prompt_mean" and "prompt_group_indices" in data:
        partitions = _prompt_group_partitions(
            data["prompt_group_indices"],
            total_lengths,
            dp_size,
            balance_data=args.balance_data,
        )
    elif args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    rollout_data_refs = []

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
        rollout_data_refs.append(Box(ray.put(rollout_data)))
    return rollout_data_refs
