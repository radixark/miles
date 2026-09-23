import asyncio
import threading
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.broadcast import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
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
                Namespace(),
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

    def test_an_engine_that_refuses_the_group_fails_the_connect(self) -> None:
        """The submitted joins are awaited, so a refusing engine surfaces instead of being dropped."""
        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray_mock,
            patch(f"{_BROADCAST_MODULE}.init_process_group", return_value=MagicMock(name="nccl_group")),
        ):
            ray_mock._private.services.get_node_ip_address.return_value = "10.0.0.1"
            with pytest.raises(RuntimeError, match="engine refused the group"):
                connect_rollout_engines_from_distributed(
                    Namespace(rollout_num_gpus_per_engine=2),
                    "miles-pp_0",
                    [_AcceptingEngine(), _AcceptingEngine(), _RefusingEngine()],
                )


class TestDisconnectRolloutEnginesFromDistributed:
    def test_disconnect_awaits_engine_cleanup_when_local_destroy_fails(self) -> None:
        """A failed local teardown still surfaces, and it must not abandon the engine-side teardown."""
        engines = [_SlowTeardownEngine(), _SlowTeardownEngine()]

        with patch(f"{_BROADCAST_MODULE}.dist") as dist_mock:
            dist_mock.destroy_process_group.side_effect = RuntimeError("nccl teardown failed")
            with pytest.raises(RuntimeError, match="nccl teardown failed"):
                disconnect_rollout_engines_from_distributed(
                    Namespace(),
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
