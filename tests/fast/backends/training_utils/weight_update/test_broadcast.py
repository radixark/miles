import asyncio
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config

from miles.backends.training_utils.weight_update.protocols.broadcast import (
    UpdateWeightFromDistributed,
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    pp_connect_pause_mode,
    update_weights_from_distributed,
)
from miles.utils import async_utils

_BROADCAST_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"

_BASE_NAMED_TENSORS = [
    ("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, 3)),
    ("model.layers.0.mlp.up_proj.weight", torch.arange(6, dtype=torch.float32).reshape(2, 3).t()),
]


class _GatedEngine:
    def __init__(self, started: threading.Semaphore, release: threading.Event) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._started = started
        self._release = release

    def __getattr__(self, name: str):
        async def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            self._started.release()
            released = await asyncio.get_running_loop().run_in_executor(None, self._release.wait, 5)
            if not released:
                raise AssertionError("the caller waited for an engine before reaching the collective")
            return {"success": True}

        return method


class _AcceptingEngine:
    async def init_weights_update_group(self, *args, **kwargs) -> dict:
        return {"success": True}


class _RecordingEngine:
    def __init__(self, calls: list[tuple[int, str]], index: int, *, refuse: bool = False) -> None:
        self._calls, self._index, self._refuse = calls, index, refuse

    async def pause_generation(self, mode: str) -> None:
        self._calls.append((self._index, f"pause_generation:{mode}"))

    async def init_weights_update_group(self, *args, **kwargs) -> dict:
        self._calls.append((self._index, "init_weights_update_group"))
        if self._refuse:
            raise RuntimeError("engine refused the group")
        return {"success": True}

    async def continue_generation(self) -> None:
        self._calls.append((self._index, "continue_generation"))


def _pause_args(*, pp_size: int | None, fully_async: bool = False, colocate: bool = False) -> Namespace:
    return Namespace(
        sglang=make_sglang_config(pp_size=pp_size),
        pause_generation_mode="retract",
        fully_async=fully_async,
        colocate=colocate,
    )


class _RefusingEngine:
    async def init_weights_update_group(self, *args, **kwargs) -> None:
        raise RuntimeError("engine refused the group")


class _SlowTeardownEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.finished = False

    async def destroy_weights_update_group(self, group_name: str) -> dict:
        self.calls.append(("destroy_weights_update_group", group_name))
        await asyncio.sleep(0.05)
        self.finished = True
        return {"success": True}


class _RecordingHandle:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class TestConnectRolloutEnginesFromDistributed:
    def test_group_init_starts_all_engines_before_local_join_with_heterogeneous_offsets(self) -> None:
        """Every engine is asked to join, at its own rank offset, before rank zero blocks on the handshake."""
        started = threading.Semaphore(0)
        release = threading.Event()
        engines = [_GatedEngine(started, release) for _ in range(3)]
        group = MagicMock(name="nccl_group")

        def join(**kwargs):
            for _ in engines:
                assert started.acquire(timeout=30), "an engine had not been asked before the local join"
            release.set()
            return group

        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", side_effect=join) as init_process_group,
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            result = connect_rollout_engines_from_distributed(
                "miles-pp_0",
                engines,
                engine_gpu_counts=[2, 4, 1],
            )

        assert result is group
        master_port = engines[0].calls[0][1][1]
        assert [engine.calls for engine in engines] == [
            [("init_weights_update_group", ("10.0.0.1", master_port, 1, 8, "miles-pp_0"), {"backend": "nccl"})],
            [("init_weights_update_group", ("10.0.0.1", master_port, 3, 8, "miles-pp_0"), {"backend": "nccl"})],
            [("init_weights_update_group", ("10.0.0.1", master_port, 7, 8, "miles-pp_0"), {"backend": "nccl"})],
        ]
        assert init_process_group.call_args.kwargs == {
            "backend": "nccl",
            "init_method": f"tcp://10.0.0.1:{master_port}",
            "world_size": 8,
            "rank": 0,
            "group_name": "miles-pp_0",
        }

    def test_missing_gpu_counts_are_rejected_before_any_engine_is_asked(self) -> None:
        """The inference controller always reports per-engine GPU counts, so a caller without them is a bug."""
        engines = [MagicMock(spec=_AcceptingEngine) for _ in range(2)]

        with (
            patch(f"{_BROADCAST_MODULE}.init_process_group") as init_process_group,
            pytest.raises(AssertionError),
        ):
            connect_rollout_engines_from_distributed("miles-pp_0", engines)

        init_process_group.assert_not_called()
        for engine in engines:
            engine.init_weights_update_group.assert_not_called()

    def test_an_engine_that_refuses_the_group_fails_the_connect(self) -> None:
        """The submitted joins are awaited, so a refusing engine surfaces instead of being dropped."""
        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", return_value=MagicMock(name="nccl_group")),
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            with pytest.raises(RuntimeError, match="engine refused the group"):
                connect_rollout_engines_from_distributed(
                    "miles-pp_0",
                    [_AcceptingEngine(), _AcceptingEngine(), _RefusingEngine()],
                    engine_gpu_counts=[2, 2, 2],
                )


class TestPipelineParallelEnginesArePausedForTheGroupInit:
    @pytest.mark.parametrize(
        "args,expected",
        [
            (_pause_args(pp_size=None), None),
            (_pause_args(pp_size=1), None),
            (_pause_args(pp_size=2), "retract"),
            # The driver already paused them for the whole update.
            (_pause_args(pp_size=2, fully_async=True, colocate=True), None),
        ],
    )
    def test_only_pipeline_parallel_engines_the_driver_has_not_paused_are_paused(self, args, expected) -> None:
        assert pp_connect_pause_mode(args) == expected

    def test_every_engine_is_paused_before_any_joins_and_resumed_after(self) -> None:
        """A PP stage blocked in the group init stops relaying micro-batches the next stage is waiting for."""
        calls: list[tuple[int, str]] = []
        engines = [_RecordingEngine(calls, index) for index in range(2)]

        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", return_value=MagicMock(name="nccl_group")),
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            connect_rollout_engines_from_distributed(
                "miles-pp_0", engines, engine_gpu_counts=[2, 2], pause_generation_mode="retract"
            )

        steps = [step for _, step in calls]
        assert steps[:2] == ["pause_generation:retract"] * 2
        assert sorted(steps[2:4]) == ["init_weights_update_group"] * 2
        assert steps[4:] == ["continue_generation"] * 2

    def test_engines_are_resumed_when_the_group_init_fails(self) -> None:
        calls: list[tuple[int, str]] = []
        engines = [_RecordingEngine(calls, 0), _RecordingEngine(calls, 1, refuse=True)]

        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", return_value=MagicMock(name="nccl_group")),
            pytest.raises(RuntimeError, match="engine refused the group"),
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            connect_rollout_engines_from_distributed(
                "miles-pp_0", engines, engine_gpu_counts=[1, 1], pause_generation_mode="retract"
            )

        assert sorted(index for index, step in calls if step == "continue_generation") == [0, 1]

    def test_engines_without_pipeline_parallelism_are_not_paused(self) -> None:
        calls: list[tuple[int, str]] = []
        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", return_value=MagicMock(name="nccl_group")),
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            connect_rollout_engines_from_distributed("miles-pp_0", [_RecordingEngine(calls, 0)], engine_gpu_counts=[1])

        assert calls == [(0, "init_weights_update_group")]


class TestUpdateWeightFromDistributedConnect:
    def test_the_protocol_forwards_discovered_gpu_counts_to_the_group_builder(self) -> None:
        """The protocol must not replace heterogeneous discovered counts with the uniform fallback."""
        initial_parallel_state = SimpleNamespace(
            pp=SimpleNamespace(size=1, rank=0),
            tp=SimpleNamespace(rank=0),
            intra_dp_cp=SimpleNamespace(rank=0),
        )
        with patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=initial_parallel_state):
            protocol = UpdateWeightFromDistributed(
                Namespace(check_weight_update_equal=False, **vars(_pause_args(pp_size=1)))
            )

        engines = [MagicMock(name="prefill"), MagicMock(name="decode")]
        parallel_state = SimpleNamespace(pp=SimpleNamespace(rank=0))
        with (
            patch(f"{_BROADCAST_MODULE}.get_data_replica_rank_and_size", return_value=(0, 1)),
            patch(f"{_BROADCAST_MODULE}.disconnect_rollout_engines_from_distributed"),
            patch(f"{_BROADCAST_MODULE}.connect_rollout_engines_from_distributed") as connect,
        ):
            protocol.connect(
                engines,
                engine_gpu_counts=[2, 4],
                engine_gpu_offsets=[0, 2],
                engine_cell_ids=["cell-0", "cell-1"],
                parallel_state=parallel_state,
                placement=SimpleNamespace(gather_pp=False),
                selector="all",
            )

        connect.assert_called_once_with(
            "miles-pp_0",
            engines,
            engine_gpu_counts=[2, 4],
            pause_generation_mode=None,
        )


class TestDisconnectRolloutEnginesFromDistributed:
    def test_disconnect_awaits_engine_cleanup_when_local_destroy_fails(self) -> None:
        """A failed local teardown still surfaces, and it must not abandon the engine-side teardown."""
        engines = [_SlowTeardownEngine(), _SlowTeardownEngine()]

        with patch(f"{_BROADCAST_MODULE}.dist") as dist_mock:
            dist_mock.destroy_process_group.side_effect = RuntimeError("nccl teardown failed")
            with pytest.raises(RuntimeError, match="nccl teardown failed"):
                disconnect_rollout_engines_from_distributed(
                    "miles-pp_0",
                    MagicMock(name="nccl_group"),
                    engines,
                )

        assert [engine.calls for engine in engines] == [
            [("destroy_weights_update_group", "miles-pp_0")],
            [("destroy_weights_update_group", "miles-pp_0")],
        ]
        assert [engine.finished for engine in engines] == [True, True]


class TestUpdateWeightsFromDistributed:
    def test_base_metadata_is_in_flight_before_broadcast_and_all_handles_finish(self) -> None:
        """No engine is still unasked when the first tensor goes out, every tensor is staged contiguously, and no handle is left unwaited."""
        started = threading.Semaphore(0)
        release = threading.Event()
        engines = [_GatedEngine(started, release) for _ in range(3)]
        handles: list[_RecordingHandle] = []
        broadcast_tensors: list[torch.Tensor] = []

        def broadcast(tensor, src, group=None, async_op=False):
            broadcast_tensors.append(tensor)
            if len(broadcast_tensors) == 1:
                for _ in engines:
                    assert started.acquire(timeout=30), "an engine had not been asked before the first broadcast"
                release.set()
            handles.append(_RecordingHandle())
            return handles[-1]

        with patch(f"{_BROADCAST_MODULE}.dist") as dist_mock:
            dist_mock.broadcast.side_effect = broadcast
            futures = update_weights_from_distributed(
                "miles-pp_0",
                MagicMock(name="nccl_group"),
                engines,
                _BASE_NAMED_TENSORS,
                selector="target",
            )
            async_utils.wait_futures(futures)

        assert [len(engine.calls) for engine in engines] == [1, 1, 1]
        assert engines[0].calls[0][2]["names"] == [name for name, _ in _BASE_NAMED_TENSORS]
        assert engines[0].calls[0][2]["selector"] == "target"
        assert [handle.waited for handle in handles] == [True, True]
        assert [tensor.is_contiguous() for tensor in broadcast_tensors] == [True, True]
        assert torch.equal(broadcast_tensors[1], _BASE_NAMED_TENSORS[1][1])


class TestUpdateWeightFromDistributedLoraBaseResync:
    @pytest.mark.parametrize("checker", [True, False])
    def test_the_weight_checker_forces_a_base_resync(self, checker: bool) -> None:
        """The checker scrambles the engine base before the first sync, so a LoRA sync must resend it."""
        parallel_state = SimpleNamespace(
            pp=SimpleNamespace(size=1, rank=0),
            tp=SimpleNamespace(rank=0),
            intra_dp_cp=SimpleNamespace(rank=0),
        )
        with patch(f"{_BROADCAST_MODULE}.get_parallel_state", return_value=parallel_state):
            protocol = UpdateWeightFromDistributed(Namespace(check_weight_update_equal=checker))

        assert protocol.needs_base_resync_for_lora is checker
