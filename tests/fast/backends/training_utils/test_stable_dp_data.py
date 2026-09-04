from argparse import Namespace
from types import SimpleNamespace

from miles.backends.training_utils import data as data_module


def test_degraded_retry_preserves_nominal_strided_dp_partitions(monkeypatch) -> None:
    """A full-degraded retry schedules target samples in the original logical DP shards."""
    parallel_state = SimpleNamespace(
        effective_dp=SimpleNamespace(size=1, rank=0, group=None),
        cp=SimpleNamespace(size=1),
        vpp_size=1,
        microbatch_group_size_per_vp_stage=None,
    )
    monkeypatch.setattr(data_module, "get_parallel_state", lambda: parallel_state)
    args = Namespace(
        qkv_format="thd",
        use_dynamic_global_batch_size=False,
        global_batch_size=8,
        balance_data=False,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=100,
    )
    rollout_data = {
        "tokens": [[index] for index in range(8)],
        "total_lengths": [60, 20, 60, 20, 60, 20, 60, 20],
        "stable_dp_size": 2,
    }

    data_iterators, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [8]
    assert "stable_dp_size" not in rollout_data
    assert rollout_data["stable_dp_microbatches"] == [[4, 4]]
    assert data_iterators[0].micro_batch_indices == [[6], [4], [2], [0], [7], [5], [3], [1]]
