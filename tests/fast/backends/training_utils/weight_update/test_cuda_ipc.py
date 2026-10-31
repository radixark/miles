import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.cuda_ipc import (
    UpdateWeightFromTensor,
    _ipc_ref_counter,
    _send_to_colocated_engine,
)
from miles.utils import async_utils

_TENSOR_MODULE = "miles.backends.training_utils.weight_update.protocols.cuda_ipc"

_ENGINE_GPU_COUNTS = [2, 2, 2, 2]
_SPARSE_GPU_OFFSETS = [0, 2, 6, 8]


class _FakeFlattenedTensorBucket:
    supports_multi_dtypes = False

    def __init__(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        self._named_tensors = named_tensors

    def get_flattened_tensor(self) -> str:
        return self._named_tensors[0][0]

    def get_metadata(self) -> None:
        return None


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(data: dict, output_str: bool = False) -> str:
        return f"serialized:{data['flattened_tensor']}"


class _SerialProbeEngine:
    def __init__(self) -> None:
        self.received: list[str] = []
        self.active = 0
        self.max_active = 0

    async def update_weights_from_tensor(self, **kwargs) -> dict:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.received.append(kwargs["serialized_named_tensors"][0])
        await asyncio.sleep(0.01)
        self.active -= 1
        return {"success": True}


def _gather_object_onto_source(obj, object_gather_list=None, dst=None, group=None) -> None:
    object_gather_list[0] = obj


class TestConnect:
    @staticmethod
    def _make_protocol() -> UpdateWeightFromTensor:
        protocol = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
        protocol.args = Namespace(
            train_backend="megatron",
            rollout_num_gpus_per_engine=2,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            check_weight_update_equal=False,
            colocate=True,
            offload_rollout=True,
            lora_sync_sends_base=True,
        )
        protocol.rollout_engines = None
        protocol._model_update_groups = None
        protocol._ipc_gather_group = MagicMock(name="gather_group_from_init")
        protocol._ipc_gather_src = 0
        return protocol

    @staticmethod
    def _connect(
        protocol: UpdateWeightFromTensor, engines: list, *, rank: int, tp_rank: int
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        parallel_state = SimpleNamespace(
            intra_dp_cp=SimpleNamespace(rank=0),
            tp=SimpleNamespace(rank=tp_rank),
            pp=SimpleNamespace(rank=0),
        )
        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.lora_rollout_enabled", return_value=False),
            patch(f"{_TENSOR_MODULE}.connect_rollout_engines_from_distributed") as connect,
            patch(f"{_TENSOR_MODULE}.disconnect_rollout_engines_from_distributed") as disconnect,
        ):
            dist_mock.get_rank.return_value = rank
            dist_mock.get_world_size.return_value = 8
            protocol.connect(
                engines,
                engine_gpu_counts=_ENGINE_GPU_COUNTS,
                engine_gpu_offsets=_SPARSE_GPU_OFFSETS,
                engine_cell_ids=[f"cell-{i}" for i in range(len(_ENGINE_GPU_COUNTS))],
                parallel_state=parallel_state,
                placement=SimpleNamespace(gather_pp=True),
                selector="all",
            )
        return dist_mock, connect, disconnect

    @pytest.mark.parametrize(
        ("rank", "expected_engine_index", "expected_gather_src"),
        [(0, 0, 0), (6, 2, 6)],
        ids=["first_engine", "engine_after_sparse_gap"],
    )
    def test_connect_splits_clients_and_maps_sparse_gpu_offsets(
        self, rank: int, expected_engine_index: int, expected_gather_src: int
    ) -> None:
        """Only remote engines use NCCL, while ranks on either side of a sparse gap map to their covered colocated engine."""
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]
        protocol = self._make_protocol()
        protocol._ipc_gather_src = expected_gather_src
        gather_group_from_init = protocol._ipc_gather_group

        dist_mock, connect, disconnect = self._connect(protocol, engines, rank=rank, tp_rank=0)

        assert protocol.use_distribute is True
        assert protocol.rollout_engines == engines[:3]
        assert protocol.distributed_rollout_engines == engines[3:]
        connect.assert_called_once_with("miles", engines[3:], engine_gpu_counts=[2])
        disconnect.assert_not_called()
        assert protocol._model_update_groups is connect.return_value
        assert protocol._ipc_engine is engines[expected_engine_index]
        assert protocol._ipc_gather_group is gather_group_from_init
        assert protocol._ipc_gather_src == expected_gather_src
        assert protocol.is_sender is True
        dist_mock.new_group.assert_not_called()

    def test_connect_clears_ipc_state_for_a_placeholder_rank(self) -> None:
        """A reserved GPU slot that no engine covers drops its stale gather group, or it would enter a collective nobody else joins."""
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]
        protocol = self._make_protocol()

        _dist_mock, connect, _disconnect = self._connect(protocol, engines, rank=4, tp_rank=1)

        assert protocol._ipc_gather_group is None
        assert protocol._ipc_gather_src is None
        assert protocol._ipc_engine is None
        assert protocol.is_sender is False
        connect.assert_not_called()


class TestColocatedBaseSend:
    def test_bucket_storage_is_released_after_all_ipc_consumers_finish(self) -> None:
        protocol = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
        protocol.use_distribute = False
        protocol._ipc_engine = MagicMock()
        protocol._ipc_gather_src = 0
        protocol._ipc_gather_group = MagicMock(name="ipc_group")
        protocol._selector = "all"
        protocol._ring = None  # default path: one fresh CUDA-IPC bucket per send
        future = MagicMock(name="engine_future")

        with (
            patch(f"{_TENSOR_MODULE}._send_to_colocated_engine", return_value=([future], object(), ["payload"])),
            patch(f"{_TENSOR_MODULE}.async_utils.wait_futures", return_value=[{"success": True}]) as wait,
            patch(f"{_TENSOR_MODULE}.check_weight_sync_results") as check,
            patch(f"{_TENSOR_MODULE}.dist.barrier") as barrier,
        ):
            protocol.send_bucket([("weight", torch.zeros(1))])

        wait.assert_called_once_with([future])
        check.assert_called_once_with([{"success": True}], is_lora=False)
        barrier.assert_called_once_with(group=protocol._ipc_gather_group)

    def test_multiple_colocated_buckets_are_sent_serially_in_source_order(self) -> None:
        """Two dtype buckets reach the engine one at a time and in the order they were built."""
        engine = _SerialProbeEngine()
        named_tensors = [
            ("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, dtype=torch.float32)),
            ("model.layers.0.mlp.up_proj.weight", torch.zeros(2, dtype=torch.float16)),
        ]

        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.FlattenedTensorBucket", _FakeFlattenedTensorBucket),
            patch(f"{_TENSOR_MODULE}.MultiprocessingSerializer", _FakeMultiprocessingSerializer),
        ):
            dist_mock.get_rank.return_value = 0
            dist_mock.get_world_size.return_value = 1
            dist_mock.gather_object.side_effect = _gather_object_onto_source
            futures, _long_lived_tensors, _serialized = _send_to_colocated_engine(
                named_tensors,
                ipc_engine=engine,
                ipc_gather_src=0,
                ipc_gather_group=MagicMock(name="gather_group"),
            )
            async_utils.wait_futures(futures)

        assert engine.received == [
            "serialized:model.layers.0.mlp.gate_proj.weight",
            "serialized:model.layers.0.mlp.up_proj.weight",
        ]
        assert engine.max_active == 1


class TestIpcBucketRing:
    def test_ref_counter_is_read_from_the_reducer_pickle(self) -> None:
        """The debug reader finds torch's ref-counter shm name and slot offset without unpickling (no IPC open)."""
        import base64
        import pickle

        # same positional layout as torch.multiprocessing.reductions.reduce_tensor's args tuple
        args = (
            torch.Tensor,
            torch.Size([5]),
            (1,),
            0,
            torch.storage.TypedStorage,
            torch.uint8,
            "GPU-0123",
            b"\x01" * 64,
            4096,
            0,
            False,
            b"/torch_4242_1518968791_17",
            9,
            b"\x02" * 64,
            False,
        )
        payload = {"flattened_tensor": ("rebuild", args), "metadata": [("b0.t0", "/torch_1_1 is only a name here", 3)]}
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        assert _ipc_ref_counter(data) == ("/torch_4242_1518968791_17", 9)
        assert _ipc_ref_counter(base64.b64encode(data).decode()) == ("/torch_4242_1518968791_17", 9)
        assert _ipc_ref_counter(pickle.dumps({"metadata": []})) is None

    def test_ring_handshake_shares_the_engine_verdict_with_every_rank(self) -> None:
        """With the ring, a non-source rank waits for the gather-source rank's verdict and raises on an engine error
        instead of running ahead into the next bucket."""
        protocol = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
        protocol.args = Namespace(update_weight_buffer_size=16)
        protocol.use_distribute = False
        protocol._ring = object()  # any ring: only the handshake path is exercised
        protocol._ring_debug_left = 0
        protocol._ipc_gather_src = 0
        protocol._ipc_gather_group = MagicMock(name="gather_group")
        protocol._ipc_engine = None
        protocol._selector = "all"

        def all_gather(obj_list, obj, group):
            assert group is protocol._ipc_gather_group
            assert obj is None  # this rank holds no futures, so it has no error of its own
            assert len(obj_list) == 8
            obj_list[0] = "rank 0: RuntimeError: engine said no"

        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}._send_to_colocated_engine", return_value=([], [], ["payload"])),
        ):
            dist_mock.get_rank.return_value = 3  # not the gather source: holds no futures
            dist_mock.get_world_size.return_value = 8
            dist_mock.all_gather_object.side_effect = all_gather
            with pytest.raises(RuntimeError, match="engine said no"):
                protocol.send_bucket([("w", torch.zeros(1))])
            dist_mock.all_gather_object.assert_called_once()
            dist_mock.barrier.assert_not_called()


class TestIpcBucketRingStage:
    """CPU-only: slot rotation, growth, cached share args and the re-share fallback of ``_IpcBucketRing.stage``."""

    @staticmethod
    def _fake_share(buf: torch.Tensor):
        # positional layout of torch.multiprocessing.reductions.reduce_tensor's args
        return (
            "rebuild",
            (
                torch.Tensor,
                torch.Size([buf.numel()]),
                (1,),
                0,
                torch.storage.TypedStorage,
                buf.dtype,
                "GPU-uuid",
                b"\x01" * 64,
                buf.numel(),
                0,
                False,
                b"/torch_1_2_0",
                7,
                b"\x02" * 64,
                True,
            ),
        )

    def _ring(self, slots: int, min_bytes: int, reshare: bool = False):
        from miles.backends.training_utils.weight_update.protocols.cuda_ipc import _IpcBucketRing

        ring = _IpcBucketRing(slots, min_bytes=min_bytes, reshare=reshare)

        def fake_share(buf: torch.Tensor):
            # mirror the real _share_once: it clears event_sync_required (rule 1) and records the ref-counter file
            fn, args = self._fake_share(buf)
            args = list(args)
            args[_IpcBucketRing._ARG_EVENT_SYNC] = False
            ring._note_counter_file(args[_IpcBucketRing._ARG_REF_COUNTER_HANDLE])
            return fn, tuple(args)

        ring._share_once = fake_share  # type: ignore[method-assign]
        return ring

    def test_slots_rotate_and_the_cached_share_carries_the_bucket_length(self) -> None:
        from miles.backends.training_utils.weight_update.protocols.cuda_ipc import _SharedTensorRef

        ring = self._ring(2, min_bytes=16)
        stream = MagicMock(name="stream")
        with patch("torch.cuda.current_stream", return_value=stream):
            refs = [ring.stage(torch.full((n,), i, dtype=torch.uint8)) for i, n in enumerate((4, 6, 8))]
        assert all(isinstance(r, _SharedTensorRef) for r in refs)
        assert [r.args[1] for r in refs] == [torch.Size([4]), torch.Size([6]), torch.Size([8])]
        assert all(r.args[2] == (1,) and r.args[3] == 0 and r.args[14] is False for r in refs)
        assert refs[0].args[7] == refs[2].args[7]  # bucket 2 reused slot 0's one-time share
        assert stream.synchronize.call_count == 3
        slots = ring._slots[(torch.uint8, torch.device("cpu"))]
        assert len(slots) == 2 and ring.reallocs == 2 and ring.retired == []
        assert slots[0].buf[:8].tolist() == [2] * 8  # bucket 2 overwrote slot 0 with its bytes
        assert ring.counter_files == {"/dev/shm/torch_1_2_0"}

    def test_oversized_bucket_grows_the_slot_and_retires_the_old_buffer(self) -> None:
        ring = self._ring(2, min_bytes=8)
        with patch("torch.cuda.current_stream", return_value=MagicMock()):
            ring.stage(torch.zeros(8, dtype=torch.uint8))  # slot 0 at min size
            ring.stage(torch.zeros(8, dtype=torch.uint8))  # slot 1
            big = ring.stage(torch.ones(32, dtype=torch.uint8))  # slot 0 again: too small -> grow
        assert ring.reallocs == 3 and len(ring.retired) == 1 and ring.retired[0].numel() == 8
        assert big.args[1] == torch.Size([32])
        assert ring._slots[(torch.uint8, torch.device("cpu"))][0].buf.numel() == 32

    def test_reshare_mode_returns_the_slot_view_itself(self) -> None:
        ring = self._ring(1, min_bytes=16, reshare=True)
        with patch("torch.cuda.current_stream", return_value=MagicMock()):
            view = ring.stage(torch.arange(5, dtype=torch.uint8))
        assert isinstance(view, torch.Tensor) and view.numel() == 5 and view.tolist() == [0, 1, 2, 3, 4]
        assert ring._slots[(torch.uint8, torch.device("cpu"))][0].shared is None


class TestObservedWorldSize:
    def test_colocation_uses_the_live_world_instead_of_declared_actor_capacity(self) -> None:
        """A smaller live trainer world sends engines outside that world through distributed transfer."""
        protocol = TestConnect._make_protocol()
        protocol.args.actor_num_gpus_per_node = 32
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]

        _, connect, _ = TestConnect._connect(protocol, engines, rank=0, tp_rank=0)

        assert protocol.rollout_engines == engines[:3]
        assert protocol.distributed_rollout_engines == engines[3:]
        connect.assert_called_once_with("miles", engines[3:], engine_gpu_counts=[2])
