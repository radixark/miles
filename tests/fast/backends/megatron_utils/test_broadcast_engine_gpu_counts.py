from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast import (
    UpdateWeightFromDistributed,
    connect_rollout_engines_from_distributed,
)

_BROADCAST_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"


class _RecordingEngine:
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def init_weights_update_group(self, master_address, master_port, rank, world_size, group_name, backend):
        self._calls.append(dict(rank=rank, world_size=world_size))
        return None


def _connect(*, num_engines: int, engine_gpu_counts, rollout_num_gpus_per_engine: int) -> tuple[list[dict], MagicMock]:
    calls: list[dict] = []
    engines = [_RecordingEngine(calls) for _ in range(num_engines)]
    args = Namespace(rollout_num_gpus_per_engine=rollout_num_gpus_per_engine)
    async_utils = SimpleNamespace(submit=lambda coro: coro, wait_futures=lambda futures: None)

    with (
        patch(f"{_BROADCAST_MODULE}.ray"),
        patch(f"{_BROADCAST_MODULE}.async_utils", async_utils),
        patch(f"{_BROADCAST_MODULE}.init_process_group") as init_group,
    ):
        connect_rollout_engines_from_distributed(args, "miles-pp_0", engines, engine_gpu_counts=engine_gpu_counts)
    return calls, init_group


class TestConnectRolloutEnginesFromDistributed:
    def test_heterogeneous_engines_each_take_as_many_ranks_as_they_have_gpus(self):
        """External PD runs prefill and decode at different tp sizes; sizing the group from a single
        per-engine constant makes init_weights_update_group and init_process_group disagree, and NCCL
        then waits for a rank that never joins, with no timeout."""
        calls, init_group = _connect(num_engines=2, engine_gpu_counts=[2, 4], rollout_num_gpus_per_engine=1)

        assert [call["rank"] for call in calls] == [1, 3]
        assert {call["world_size"] for call in calls} == {7}
        assert init_group.call_args.kwargs["world_size"] == 7

    def test_without_counts_the_group_falls_back_to_the_uniform_argument(self):
        """Callers that know nothing about per-engine sizes keep the old uniform layout."""
        calls, init_group = _connect(num_engines=3, engine_gpu_counts=None, rollout_num_gpus_per_engine=2)

        assert [call["rank"] for call in calls] == [1, 3, 5]
        assert init_group.call_args.kwargs["world_size"] == 7


class TestConnectRolloutEnginesForwardsTheDiscoveredCounts:
    def test_the_updater_hands_the_per_engine_counts_to_the_group_builder(self):
        """The counts arrive from engine discovery; dropping them here is what let the uniform
        fallback size an external fleet's NCCL group wrongly."""
        parallel_state = SimpleNamespace(
            pp=SimpleNamespace(size=1, rank=0),
            tp=SimpleNamespace(rank=0),
            intra_dp_cp=SimpleNamespace(rank=0),
        )
        with (
            patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=parallel_state),
            patch(f"{_BROADCAST_MODULE}.create_world_ticket_lock"),
            patch.object(UpdateWeightFromDistributed, "_init_lora"),
        ):
            updater = UpdateWeightFromDistributed(
                Namespace(), [], lambda: {}, model_name="test-model", quantization_config=None
            )

        engines = [object(), object()]
        with (
            patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=parallel_state),
            patch(f"{_BROADCAST_MODULE}.disconnect_rollout_engines_from_distributed"),
            patch(f"{_BROADCAST_MODULE}.connect_rollout_engines_from_distributed") as connect,
        ):
            updater.connect_rollout_engines(engines, engine_gpu_counts=[2, 4], engine_gpu_offsets=[0, 2])

        assert connect.call_args.kwargs["engine_gpu_counts"] == [2, 4]
