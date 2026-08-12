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
    "teacher_log_probs": "float32",
    "opd_reverse_kl": "float32",
    "rollout_routed_experts": "int32",
    "rollout_indexer_topk": "int32",
    # Experience-augmented teacher view for in-trainer top-k OPD
    # (--opd-topk-in-trainer): privileged_prefix + response tokens.
    "teacher_tokens": "int32",
    # turnhint OPD: per-sample map from student response index -> position in
    # the teacher view's own response span (identity arange when the view is
    # suffix-aligned, so the batch representation is uniform).
    "teacher_gather_positions": "int32",
}

ROLLOUT_DATA_VALUE_SPEC: dict[str, ValueSpec] = {
    **{field: ValueSpec(codec="typed_ragged") for field in ROLLOUT_DATA_TENSOR_DTYPES},
    "partition": ValueSpec(codec="ndarray", dtype="int64"),
    "seq_witness_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "response_lengths": ValueSpec(codec="ndarray", dtype="int64"),
    "teacher_response_lengths": ValueSpec(codec="ndarray", dtype="int64"),
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

    if any(sample.teacher_tokens is not None for sample in samples):
        # Experience-augmented teacher view for in-trainer OPD
        # (--opd-topk-in-trainer). The whole KL alignment contract is "the last
        # response_length positions of the teacher view are the SAME response
        # tokens the student scored" -- pin it here, fail loud, before anything
        # downstream can silently misalign.
        #
        # Gate on `any`, not `samples[0]`: a train-data conversion may produce a
        # mixed batch (some samples with a teacher view, some without) when the
        # reward post-process excludes part of the batch from teacher scoring
        # without dropping them. A sample with no teacher view falls back to its
        # own tokens: teacher == student for that sample only, i.e. zero forced
        # KL contribution from a sample the teacher never scored.
        for sample in samples:
            if sample.teacher_tokens is None:
                # copy, not alias: a later in-place mutation of tokens must not
                # silently rewrite the teacher view
                sample.teacher_tokens = list(sample.tokens)
            r = sample.response_length
            assert len(sample.teacher_tokens) >= r, (
                f"teacher_tokens shorter than response for sample {sample.index}"
            )
            if sample.teacher_gather_positions is not None:
                # turnhint view: hint turns are interleaved INSIDE the response,
                # so alignment rests on the position map -- pin that the mapped
                # positions recover exactly the student's response tokens, fail
                # loud before anything downstream can silently gather at the wrong
                # rows.
                positions = sample.teacher_gather_positions
                t_resp = sample.teacher_response_length
                assert t_resp is not None and t_resp >= r, (
                    f"teacher_response_length ({t_resp}) must cover the student "
                    f"response ({r}) for sample {sample.index}"
                )
                assert len(positions) == r, (
                    f"teacher_gather_positions maps {len(positions)} tokens but the "
                    f"student response has {r} for sample {sample.index}"
                )
                # Positions must be in-range, strictly increasing, duplicate-free:
                # a negative index would wrap and (when the tokens happen to match)
                # pass the content check below while silently selecting the wrong
                # row in the gather; a duplicate would double-count its KL term.
                assert all(0 <= p < t_resp for p in positions) and list(positions) == sorted(
                    set(positions)
                ), (
                    f"teacher_gather_positions out of range or not strictly increasing "
                    f"for sample {sample.index} (t_resp={t_resp})"
                )
                span = sample.teacher_tokens[len(sample.teacher_tokens) - t_resp :]
                assert [span[p] for p in positions] == list(
                    sample.tokens[len(sample.tokens) - r:]
                ), (
                    f"teacher view's mapped positions != student response tokens "
                    f"for sample {sample.index} -- the spliced teacher view corrupted "
                    "the student token subsequence"
                )
            else:
                assert list(sample.teacher_tokens[len(sample.teacher_tokens) - r:]) == list(
                    sample.tokens[len(sample.tokens) - r:]
                ), f"teacher_tokens response suffix != tokens response suffix for sample {sample.index}"
        train_data["teacher_tokens"] = [sample.teacher_tokens for sample in samples]
        if any(sample.teacher_gather_positions is not None for sample in samples):
            # Uniform batch representation: suffix-aligned samples get the
            # identity map so downstream row-selects need no per-sample
            # special-casing (an identity select is a semantic no-op).
            train_data["teacher_gather_positions"] = [
                sample.teacher_gather_positions
                if sample.teacher_gather_positions is not None
                else list(range(sample.response_length))
                for sample in samples
            ]
            train_data["teacher_response_lengths"] = [
                sample.teacher_response_length
                if sample.teacher_response_length is not None
                else sample.response_length
                for sample in samples
            ]

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
        # group norm
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if prompt_group_sizes is not None:
            # Multi-LoRA: groups may have heterogeneous sizes (per-adapter
            # n_samples_per_prompt), so normalize within explicit boundaries.
            assert sum(prompt_group_sizes) == len(
                raw_rewards
            ), f"prompt group sizes sum to {sum(prompt_group_sizes)}, but got {len(raw_rewards)} rewards"
            normalized_groups = []
            for group_rewards in rewards.split(prompt_group_sizes):
                centered = group_rewards - group_rewards.mean()
                if (
                    args.advantage_estimator in ["grpo", "gspo"]
                    and args.grpo_std_normalization
                    and group_rewards.numel() > 1
                ):
                    centered = centered / (group_rewards.std() + 1e-6)
                normalized_groups.append(centered)
            return raw_rewards, torch.cat(normalized_groups).tolist()
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
            "rollout_routed_experts",
            "rollout_indexer_topk",
            "prompt",
            "teacher_log_probs",
            "opd_reverse_kl",
            "teacher_tokens",
            "teacher_gather_positions",
            "teacher_response_lengths",
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
