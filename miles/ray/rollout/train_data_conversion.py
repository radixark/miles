import logging
from typing import Any

import torch

from miles.utils import object_store
from miles.utils.dp_schedule import build_dp_schedule, has_full_schedule_config
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.object_store import ValueSpec
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.timer import Timer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": "int32",
    "loss_masks": "int32",
    "rollout_log_probs": "float32",
    "rollout_sampling_mask_ids": "int32",
    "rollout_sampling_mask_offsets": "int64",
    "teacher_log_probs": "float32",
    "opd_reverse_kl": "float32",
    "rollout_routed_experts": "int32",
    "rollout_indexer_topk": "int32",
}

ROLLOUT_DATA_VALUE_SPEC: dict[str, ValueSpec] = {
    **{field: ValueSpec(codec="typed_ragged") for field in ROLLOUT_DATA_TENSOR_DTYPES},
    "partition": ValueSpec(codec="ndarray", dtype="int64"),
    "seq_witness_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "response_lengths": ValueSpec(codec="ndarray", dtype="int64"),
    "rewards": ValueSpec(codec="ndarray", dtype="float32"),
    "truncated": ValueSpec(codec="ndarray", dtype="int64"),
    "round_number": ValueSpec(codec="ndarray", dtype="int64"),
    "sample_indices": ValueSpec(codec="ndarray", dtype="int64"),
    "rollout_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "rollout_mask_sums": ValueSpec(codec="ndarray", dtype="int64"),
    "multimodal_train_inputs": ValueSpec(codec="ragged_tensor_dict"),
    "prompt": ValueSpec(codec="msgpack_ragged"),
    "metadata": ValueSpec(codec="msgpack_ragged"),
    "weight_versions": ValueSpec(codec="msgpack_ragged"),
    "start_weight_version": ValueSpec(codec="msgpack_ragged"),
    "raw_reward": ValueSpec(codec="auto"),
    "total_lengths": ValueSpec(codec="auto"),
    "dynamic_global_batch_size": ValueSpec(codec="auto"),
    "num_microbatches": ValueSpec(codec="auto"),
    "micro_batch_indices": ValueSpec(codec="auto"),
    "num_rollouts": ValueSpec(codec="auto"),
}


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
        args,
        samples,
        custom_reward_post_process_func=custom_reward_post_process_func,
        prompt_group_sizes=metadata.get("prompt_group_sizes"),
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
        "rollout_ids": [s.rollout_id if s.rollout_id is not None else s.index for s in samples],
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

    train_data["rollout_mask_sums"] = _compute_rollout_mask_sums(train_data["rollout_ids"], loss_masks)

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    has_sampling_mask = any(sample.rollout_sampling_mask is not None for sample in samples)
    if has_sampling_mask:
        sampling_mask_ids = []
        sampling_mask_offsets = []
        for position, sample in enumerate(samples):
            sample.validate()
            if sample.rollout_sampling_mask is None:
                raise ValueError(
                    "sampling-mask data must be present for every training sample; "
                    f"missing at position={position}, sample_index={sample.index}, status={sample.status}"
                )
            ids, offsets = sample.rollout_sampling_mask._as_tensors()

            sampling_mask_ids.append(ids)
            sampling_mask_offsets.append(offsets)

        train_data["rollout_sampling_mask_ids"] = sampling_mask_ids
        train_data["rollout_sampling_mask_offsets"] = sampling_mask_offsets

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

    if any(sample.adapter is not None for sample in samples):
        assert all(sample.adapter is not None for sample in samples), "Cannot mix adapter and adapter-less samples"
        train_data["adapter_slots"] = [sample.adapter.slot for sample in samples]
        # Slots whose adapter batch completes with this batch: the trainer scales their
        # accumulated gradients by 1/adapter-batch-size and advances the LR schedule.
        step_slots = sorted(metadata.get("step_slots", []))
        train_data["step_slots"] = step_slots
        train_data["step_adapter_names"] = sorted(metadata.get("step_adapter_names", []))
        step_slot_set = set(step_slots)
        train_data["step_adapter_batch_sizes"] = {
            sample.adapter.slot: sample.metadata["adapter_global_batch_size"]
            for sample in samples
            if sample.adapter.slot in step_slot_set
        }

    if (prompt_group_sizes := metadata.get("prompt_group_sizes")) is not None:
        train_data["prompt_group_sizes"] = prompt_group_sizes

    if samples[0].opd_reverse_kl is not None:
        train_data["opd_reverse_kl"] = [sample.opd_reverse_kl for sample in samples]

    x = metadata.get("dynamic_global_batch_size")
    assert args.use_dynamic_global_batch_size == (x is not None)
    if x is not None:
        train_data["dynamic_global_batch_size"] = x

    return train_data


def _compute_rollout_mask_sums(rollout_ids: list[int], loss_masks: list[list[int]]) -> list[int]:
    """Whole-rollout loss-mask total per sample: every sibling of one rollout carries
    the sum over all of that rollout's samples, so the loss reducer reconstructs one
    token-weighted mean per rollout even when siblings land in different micro-batches."""
    totals: dict[int, int] = {}
    for rid, mask in zip(rollout_ids, loss_masks, strict=True):
        totals[rid] = totals.get(rid, 0) + sum(mask)
    return [totals[rid] for rid in rollout_ids]


def _reward_group_segments(args: Any, samples: list[Sample], prompt_group_sizes: list[int] | None) -> list[list[int]]:
    """Return the flattened row indices for each prompt reward group."""
    # Multi-LoRA records explicit prompt boundaries before flattening.
    if prompt_group_sizes is not None:
        assert sum(prompt_group_sizes) == len(
            samples
        ), f"prompt group sizes sum to {sum(prompt_group_sizes)}, but got {len(samples)} rewards"
        groups: list[list[int]] = []
        start = 0
        for size in prompt_group_sizes:
            end = start + size
            if size > 0:
                groups.append(list(range(start, end)))
            start = end
        return groups

    # Standard rollout samples carry their prompt identity in `group_index`.
    group_indices = [sample.group_index for sample in samples]
    if all(group_index is not None for group_index in group_indices):
        segments_by_group_index: dict[int, list[int]] = {}
        for segment_index, group_index in enumerate(group_indices):
            segments_by_group_index.setdefault(int(group_index), []).append(segment_index)
        return list(segments_by_group_index.values())

    # Legacy fixed-fanout batches store each prompt's segments contiguously.
    expected_samples = args.n_samples_per_prompt * args.rollout_batch_size
    if len(samples) == expected_samples:
        return [
            list(range(start, start + args.n_samples_per_prompt))
            for start in range(0, len(samples), args.n_samples_per_prompt)
        ]
    # Without prompt identities or a complete fixed layout, use one reward group.
    return [list(range(len(samples)))]


def _normalize_rewards_by_rollout(
    args: Any,
    samples: list[Sample],
    raw_rewards: list[float],
    prompt_group_sizes: list[int] | None,
) -> list[float]:
    """Normalize one shared reward per rollout, then broadcast it to siblings."""
    if not samples:
        return []

    normalized_rewards = torch.empty(len(raw_rewards), dtype=torch.float)
    for prompt_segments in _reward_group_segments(args, samples, prompt_group_sizes):
        segments_by_rollout_key: dict[int | tuple[str, int], list[int]] = {}
        for segment_index in prompt_segments:
            sample = samples[segment_index]
            if sample.rollout_id is not None:
                rollout_key = sample.rollout_id
            elif sample.index is not None:
                rollout_key = sample.index
            else:
                rollout_key = ("row", segment_index)
            segments_by_rollout_key.setdefault(rollout_key, []).append(segment_index)

        rollout_segment_groups = list(segments_by_rollout_key.items())
        shared_rewards: list[float] = []
        for rollout_key, rollout_segments in rollout_segment_groups:
            sibling_rewards = [raw_rewards[segment_index] for segment_index in rollout_segments]
            if any(reward != sibling_rewards[0] for reward in sibling_rewards[1:]):
                raise ValueError(
                    f"all samples in rollout {rollout_key!r} must share one reward; "
                    f"rows {rollout_segments} have rewards {sibling_rewards}"
                )
            shared_rewards.append(sibling_rewards[0])

        rollout_rewards = torch.tensor(shared_rewards, dtype=torch.float)
        normalized_rollout_rewards = rollout_rewards - rollout_rewards.mean()
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization and len(rollout_rewards) > 1:
            rollout_std = rollout_rewards.std()
            if rollout_std > 0:
                normalized_rollout_rewards = normalized_rollout_rewards / (rollout_std + 1e-6)

        for (_, rollout_segments), normalized_reward in zip(
            rollout_segment_groups, normalized_rollout_rewards.tolist(), strict=True
        ):
            for segment_index in rollout_segments:
                normalized_rewards[segment_index] = normalized_reward

    return normalized_rewards.tolist()


def _post_process_rewards(
    args,
    samples: list[Sample] | list[list[Sample]],
    custom_reward_post_process_func,
    prompt_group_sizes: list[int] | None = None,
):
    if (f := custom_reward_post_process_func) is not None:
        return f(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        normalized_rewards = _normalize_rewards_by_rollout(args, samples, raw_rewards, prompt_group_sizes)
        return raw_rewards, normalized_rewards

    return raw_rewards, raw_rewards


def split_train_data_by_dp(args, data: dict[str, Any], train_parallel_config: dict | None):
    """Split the train data across DP ranks and put the shards into the object store.

    When the training backend can consume a rollout-side schedule, the shards
    also carry the precomputed micro-batch layout; otherwise this falls back to
    the legacy split (the training side schedules locally)."""
    if can_schedule_on_rollout_side(args, data, train_parallel_config):
        shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=train_parallel_config)
    else:
        shards = split_train_data_by_dp_raw(args, data, dp_size=train_parallel_config["dp_size"])
    store = object_store.get_instance()
    return [store.put(value=shard, value_spec=ROLLOUT_DATA_VALUE_SPEC) for shard in shards]


def can_schedule_on_rollout_side(args, data: dict[str, Any], train_parallel_config: dict | None) -> bool:
    """Whether the rollout side can precompute the full DP/mbs schedule."""
    if not has_full_schedule_config(train_parallel_config):
        return False
    if is_multi_lora_enabled(args):
        return False
    if "multimodal_train_inputs" in data:
        return False
    if "rollout_ids" not in data:
        return False
    global_batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
    return len(set(data["rollout_ids"])) >= global_batch_size


def split_train_data_by_dp_scheduled_raw(
    args, data: dict[str, Any], *, train_parallel_config: dict
) -> list[dict[str, Any]]:
    """DP split with the micro-batch schedule precomputed on the rollout side."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    global_batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
    partitions, micro_batch_indices, num_microbatches, num_rollouts = build_dp_schedule(
        args,
        train_parallel_config,
        total_lengths,
        global_batch_size=global_batch_size,
        rollout_indices=data["rollout_ids"],
    )
    logger.info(
        f"Rollout-side DP schedule: num_samples={len(total_lengths)}, "
        f"num_rollouts={num_rollouts}, num_microbatches={num_microbatches}"
    )

    shards = _package_shards(args, data, partitions)
    for rank, shard in enumerate(shards):
        shard["num_microbatches"] = num_microbatches
        shard["micro_batch_indices"] = micro_batch_indices[rank]
        shard["num_rollouts"] = num_rollouts
    return shards


def split_train_data_by_dp_raw(args, data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    """Split the train data by data parallel size."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    # Multi-LoRA: sort partitions by adapter slot so each microbatch is
    # contiguous-by-slot (required by the per-adapter token-count math).
    adapter_slots = data.get("adapter_slots")
    if adapter_slots is not None:
        partitions = [sorted(p, key=lambda i: adapter_slots[i]) for p in partitions]

    return _package_shards(args, data, partitions)


def _package_shards(args, data: dict[str, Any], partitions) -> list[dict[str, Any]]:
    """Package one rollout_data shard per DP rank from precomputed partitions."""
    shards = []

    for i in range(len(partitions)):
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
            "rollout_ids",
            "rollout_mask_sums",
            "rollout_log_probs",
            "rollout_sampling_mask_ids",
            "rollout_sampling_mask_offsets",
            "rollout_routed_experts",
            "rollout_indexer_topk",
            "prompt",
            "teacher_log_probs",
            "opd_reverse_kl",
            "seq_witness_ids",
            "weight_versions",
            "adapter_slots",
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
            "step_slots",
            "step_adapter_names",
            "step_adapter_batch_sizes",
            "prompt_group_sizes",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        if "adapter_slots" in rollout_data:
            rollout_data["n_adapters"] = args.multi_lora_n_adapters
        shards.append(rollout_data)
    return shards


def process_rollout_data_shard(args, rollout_data):
    """Train-side completion of the DP split: drop the ``partition`` key and
    reorder the batch-global ``total_lengths`` into this shard's row order."""
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
