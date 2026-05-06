from typing import Any

from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions


def split_train_data_by_dp(args, data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    """Split the train data by data parallel size."""
    rollout_data = {}

    if "prompt" in data:
        rollout_data["prompt"] = data["prompt"]

    n_total = len(data["tokens"])

    # Trim trailing samples to dynamic_global_batch_size if it was set upstream.
    # Two sources for this key:
    #   - delay_split=False path: rollout-side _compute_dynamic_global_batch_size
    #   - delay_split=True path:  process_rollout_data computes it after ray.get
    #                             using the current (post-healing) dp_size.
    # In both cases the value is a multiple of the relevant dp_size, so trimming
    # here makes the partition exactly even across ranks.
    if "dynamic_global_batch_size" in data:
        n_kept = data["dynamic_global_batch_size"]
        assert n_kept <= n_total, (
            f"dynamic_global_batch_size={n_kept} exceeds num_samples={n_total}"
        )
        if n_kept < n_total:
            for key in list(data.keys()):
                if isinstance(data[key], list) and len(data[key]) == n_total:
                    data[key] = data[key][:n_kept]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

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
            "prompt",
            "teacher_log_probs",
            "seq_witness_ids",
        ]:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        ans.append(rollout_data)
    return ans
