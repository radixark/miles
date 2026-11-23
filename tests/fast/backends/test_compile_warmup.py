from types import SimpleNamespace

import torch

from miles.backends.torchtitan_utils.compile_warmup import warmup_shard
from miles.backends.training_utils import parallel
from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.replay.data import attach_replay_data, register_replay_list_sequential
from miles.utils.replay_base import RoutingReplayManager


def test_the_shard_is_one_step_of_the_shortest_samples():
    """Every microbatch pads to --seq-length, so the shortest samples compile the same programs as real ones."""
    shard = warmup_shard(32)

    assert shard["total_lengths"] == [2] * 32
    assert len(shard["tokens"]) == len(shard["sample_indices"]) == 32
    assert "rollout_routed_experts" not in shard


def test_the_shard_looks_like_a_rollout_of_zero_tokens():
    shard = warmup_shard(2)

    assert shard["tokens"] == [[0, 0], [0, 0]]
    assert shard["response_lengths"] == [1, 1]
    assert shard["loss_masks"] == [[1], [1]]
    assert shard["rollout_mask_sums"] == [1, 1]
    assert shard["rollout_log_probs"] == [[0.0], [0.0]]
    assert shard["total_lengths"] == [2, 2] and shard["partition"] == [0, 1]
    assert shard["rewards"] == [0.0, 0.0] and shard["sample_indices"] == [0, 1]


def test_under_rollout_routing_replay_every_moe_layer_replays_the_top_k_padding_experts(monkeypatch):
    """The shard's routing rows go through the real replay attach and route each padding token to the next top_k experts."""
    num_layers, top_k, num_experts = 3, 2, 5
    shard = warmup_shard(2, routed_experts_shape=(num_layers, top_k))
    # as prepare_rollout_data hands the shard to replay
    rollout_data = {
        "tokens": [torch.tensor(t) for t in shard["tokens"]],
        "rollout_routed_experts": [torch.from_numpy(r) for r in shard["rollout_routed_experts"]],
        "max_seq_lens": None,
    }
    args = SimpleNamespace(
        qkv_format="thd",
        allgather_cp=False,
        train_backend="torchtitan",
        data_pad_size_multiplier=1,
        save_debug_train_data=None,
    )
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0))
    monkeypatch.setattr(parallel, "_parallel_state", state)
    manager = RoutingReplayManager()
    manager.register_replay_list_func = register_replay_list_sequential
    # a model whose first layer is dense: the MoE layers' streams start at 1
    recorded = {layer: [] for layer in range(1, num_layers)}
    manager.replays = [
        SimpleNamespace(stream_idx=layer, record=rows.append, clear=rows.clear) for layer, rows in recorded.items()
    ]

    attach_replay_data(
        manager,
        args=args,
        data_iterator=[DataIterator(rollout_data, micro_batch_size=2)],
        num_microbatches=[1],
        rollout_data=rollout_data,
    )
    manager.wait_filled(0)

    for (rows,) in recorded.values():
        routed = manager.fill_replay_padding(rows, num_experts)
        assert torch.equal(routed, torch.arange(2 * 2 * top_k, dtype=routed.dtype).view(2 * 2, top_k) % num_experts)
