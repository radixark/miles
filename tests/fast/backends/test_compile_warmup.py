from miles.backends.torchtitan_utils.compile_warmup import warmup_shard


def test_the_shard_is_one_step_of_the_shortest_samples():
    """Every microbatch pads to --seq-length, so the shortest samples compile the same programs as real ones."""
    shard = warmup_shard(32)

    assert shard["total_lengths"] == [2] * 32
    assert len(shard["tokens"]) == len(shard["sample_indices"]) == 32


def test_the_shard_looks_like_a_rollout_of_zero_tokens():
    shard = warmup_shard(2)

    assert shard["tokens"] == [[0, 0], [0, 0]]
    assert shard["response_lengths"] == [1, 1]
    assert shard["loss_masks"] == [[1], [1]]
    assert shard["rollout_mask_sums"] == [1, 1]
    assert shard["rollout_log_probs"] == [[0.0], [0.0]]
    assert shard["total_lengths"] == [2, 2] and shard["partition"] == [0, 1]
    assert shard["rewards"] == [0.0, 0.0] and shard["sample_indices"] == [0, 1]
