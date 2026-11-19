"""Unit tests for the tpu-sync (Raiden) weight transfer protocol, updater integration, and Titan local export."""

import asyncio
import dataclasses
import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from miles.backends.fsdp_utils.dtensor import fsdp_dim0_block
from miles.backends.torchtitan_utils import hf_weight_iterator as titan_iter_mod
from miles.backends.torchtitan_utils.hf_weight_iterator import TitanHfWeightIterator
from miles.backends.training_utils.weight_update import updater as updater_mod
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol, get_weight_transfer_protocol
from miles.backends.training_utils.weight_update.protocols import tpu_sync
from miles.backends.training_utils.weight_update.protocols.tpu_sync import UpdateWeightTpuSync
from miles.backends.training_utils.weight_update.protocols.utils import rank_errors
from miles.backends.training_utils.weight_update.protocols.utils.rank_errors import fail_on_every_rank
from miles.backends.training_utils.weight_update.updater import WeightUpdater


@dataclasses.dataclass(frozen=True)
class _FakeRaidenId:
    job_name: str
    job_replica_id: str
    data_name: str
    data_replica_idx: int = 0


@dataclasses.dataclass
class _FakeVariableMetadataProto:
    name: str
    shape: list[int]
    mesh_shape: list[int]
    layout: list[int]
    item_size: int
    layer_idx: int
    sharding_spec: list[str]
    global_shard_indices: list[int]


class _FakeRaidenMemoryType:
    DRAM = 1


class _FakeRaidenFuture:
    """Coroutine-only future mirroring live tpu_sync: only completing when wait() is awaited."""

    def __init__(self, *, fail_error: Exception | None = None) -> None:
        self.awaited = False
        self.wait_threadsafe_called = False
        self._fail_error = fail_error

    async def wait(self) -> None:
        await asyncio.sleep(0)
        self.awaited = True
        if self._fail_error is not None:
            raise self._fail_error

    def wait_threadsafe(self) -> None:
        self.wait_threadsafe_called = True
        raise AssertionError("wait_threadsafe() must never be called; await tfut.wait() via async_utils.run()")


class _FakeController:
    def __init__(self, port: int = 0) -> None:
        self.port = port
        self.registered_units: dict[_FakeRaidenId, dict] = {}
        self.transfers: list[tuple[dict, _FakeRaidenFuture]] = []
        self.next_transfer_error: Exception | None = None

    def get_registered_units(self) -> set[_FakeRaidenId]:
        return set(self.registered_units.keys())

    def start_transfer(self, **kwargs) -> _FakeRaidenFuture:
        fut = _FakeRaidenFuture(fail_error=self.next_transfer_error)
        self.next_transfer_error = None
        self.transfers.append((kwargs, fut))
        return fut


class _FakeControllerServer:
    def __init__(self, controller: _FakeController, *, port: int = 41234, fail_on_start: bool = False) -> None:
        self.controller = controller
        self._port = port
        self._fail_on_start = fail_on_start
        self.started = False
        self.stopped = False

    def start(self) -> int:
        if self._fail_on_start:
            raise RuntimeError("bind failed")
        self.started = True
        return self._port

    def stop(self) -> None:
        self.stopped = True


class _FakeWeightSynchronizer:
    def __init__(
        self,
        device_tensors: list[list[torch.Tensor]],
        *,
        local_port: int | None = None,
        parallelism: int = 1,
        listener_port: int | None = None,
        bind_ip: str | None = None,
        unsafe_skip_buffer_lock: bool = True,
        auto_h2d: bool = False,
        base_port: int = 50000,
    ) -> None:
        self.init_kwargs = {
            "local_port": local_port,
            "parallelism": parallelism,
            "listener_port": listener_port,
            "bind_ip": bind_ip,
            "unsafe_skip_buffer_lock": unsafe_skip_buffer_lock,
            "auto_h2d": auto_h2d,
        }
        self.local_port = base_port
        self.listener_port = base_port + 1
        self.bound_batches: list[list[list[torch.Tensor]]] = [device_tensors]
        self.unbind_count = 0
        self.shutdown_count = 0
        self.d2h_count = 0

    def bind_weights(self, device_tensors: list[list[torch.Tensor]]) -> None:
        self.bound_batches.append(device_tensors)

    def unbind_weights(self) -> None:
        self.unbind_count += 1

    def d2h(self) -> None:
        self.d2h_count += 1
        raise AssertionError("d2h() must never be called before the 1st transfer completes")

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _FakeWeightSynchronizerNoUnbind:
    """Simulates a tpu_sync build without unbind_weights()/bind_weights()."""

    def __init__(self, device_tensors: list[list[torch.Tensor]], **kwargs) -> None:
        self.bound_batches = [device_tensors]
        self.local_port = 51000
        self.listener_port = 51001
        self.shutdown_count = 0

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _FakeTpuSyncEnv:
    """Installs mocked tpu_sync.* modules into sys.modules and records all calls."""

    def __init__(
        self,
        *,
        server_port: int = 41234,
        fail_server_start: bool = False,
        sync_cls: type = _FakeWeightSynchronizer,
    ) -> None:
        self.controllers: list[_FakeController] = []
        self.servers: list[_FakeControllerServer] = []
        self.synchronizers: list = []
        self.registrations: list[tuple[str, _FakeRaidenId, dict]] = []
        self._server_port = server_port
        self._fail_server_start = fail_server_start
        self._sync_cls = sync_cls

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        env = self

        def make_controller(port: int = 0) -> _FakeController:
            ctrl = _FakeController(port=port)
            env.controllers.append(ctrl)
            return ctrl

        def make_server(controller: _FakeController) -> _FakeControllerServer:
            srv = _FakeControllerServer(controller, port=env._server_port, fail_on_start=env._fail_server_start)
            env.servers.append(srv)
            return srv

        class _ClientFacade:
            def __init__(self, controller_address: str) -> None:
                self._addr = controller_address

            def register_work_unit(
                self, unit, shards, control_plane_rpc_address, mesh_shape, variables, mesh_axes
            ) -> None:
                record = {
                    "shards": shards,
                    "control_plane_rpc_address": control_plane_rpc_address,
                    "mesh_shape": mesh_shape,
                    "variables": variables,
                    "mesh_axes": mesh_axes,
                }
                env.registrations.append((self._addr, unit, record))
                if env.controllers:
                    env.controllers[-1].registered_units[unit] = record

        def make_synchronizer(device_tensors, **kwargs):
            ws = env._sync_cls(device_tensors, **kwargs)
            env.synchronizers.append(ws)
            return ws

        tpu_sync_pkg = types.ModuleType("tpu_sync")
        rpc_pkg = types.ModuleType("tpu_sync.rpc")
        raiden_controller_mod = types.ModuleType("tpu_sync.rpc.raiden_controller")
        raiden_controller_mod.RaidenController = make_controller
        raiden_controller_mod.RaidenControllerServer = make_server
        raiden_controller_mod.RaidenControllerClientFacade = _ClientFacade
        raiden_controller_mod.RaidenId = _FakeRaidenId
        raiden_controller_mod.RaidenMemoryType = _FakeRaidenMemoryType

        pb2_mod = types.ModuleType("tpu_sync.rpc.raiden_service_pb2")
        pb2_mod.VariableMetadataProto = _FakeVariableMetadataProto

        api_pkg = types.ModuleType("tpu_sync.api")
        torch_pkg = types.ModuleType("tpu_sync.api.torch")
        ws_mod = types.ModuleType("tpu_sync.api.torch.weight_synchronizer")
        ws_mod.WeightSynchronizer = make_synchronizer

        for name, mod in {
            "tpu_sync": tpu_sync_pkg,
            "tpu_sync.rpc": rpc_pkg,
            "tpu_sync.rpc.raiden_controller": raiden_controller_mod,
            "tpu_sync.rpc.raiden_service_pb2": pb2_mod,
            "tpu_sync.api": api_pkg,
            "tpu_sync.api.torch": torch_pkg,
            "tpu_sync.api.torch.weight_synchronizer": ws_mod,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)


class _RecordingRolloutClient:
    """Engine-agnostic fake client implementing contract #18 receiver RPCs."""

    def __init__(
        self,
        *,
        env: _FakeTpuSyncEnv,
        tp_size: int,
        events: list[tuple[str, str, dict]],
        engine_label: str,
    ) -> None:
        self._env = env
        self._tp_size = tp_size
        self._events = events
        self._label = engine_label

    async def init_tpu_sync_receiver(
        self,
        *,
        controller_address: str,
        job_name: str,
        weight_shapes: dict[str, list[int]],
        parallelism: int = 8,
    ) -> None:
        self._events.append(
            (
                self._label,
                "init_tpu_sync_receiver",
                {
                    "controller_address": controller_address,
                    "job_name": job_name,
                    "weight_shapes": weight_shapes,
                    "parallelism": parallelism,
                },
            )
        )
        # Simulate rollout TP workers registering themselves with the Raiden controller.
        ctrl = self._env.controllers[-1]
        for tp_rank in range(self._tp_size):
            ctrl.registered_units[
                _FakeRaidenId(job_name=job_name, job_replica_id=str(tp_rank), data_name="weights")
            ] = {}

    async def install_tpu_sync_weights(self) -> None:
        self._events.append((self._label, "install_tpu_sync_weights", {}))


def _patch_dist(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 0,
    world_size: int = 1,
    broadcast_address: str | None = None,
    other_rank_errors: list[str | None] | None = None,
    node_ip: str = "10.4.17.6",
) -> dict:
    state: dict = {"broadcasts": []}

    class _DistStub:
        @staticmethod
        def get_rank(group=None) -> int:
            return rank

        @staticmethod
        def get_world_size(group=None) -> int:
            return world_size

        @staticmethod
        def broadcast_object_list(objects: list, src: int = 0, group=None) -> None:
            assert src == 0
            if rank != 0 and broadcast_address is not None:
                objects[0] = broadcast_address
            state["broadcasts"].append(objects[0])

        @staticmethod
        def all_gather_object(output: list, value, group=None) -> None:
            if other_rank_errors is not None:
                all_vals = list(other_rank_errors)
                all_vals.insert(rank, value)
                output[:] = all_vals
            else:
                output[:] = [value if r == rank else None for r in range(world_size)]

    monkeypatch.setattr(tpu_sync, "dist", _DistStub)
    monkeypatch.setattr(tpu_sync, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(rank_errors, "dist", _DistStub)
    monkeypatch.setattr(rank_errors, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(tpu_sync, "get_current_node_ip", lambda: node_ip)
    return state


def _make_protocol(
    engines: list,
    tp_sizes: list[int],
    cell_ids: list[str] | None = None,
    *,
    ft_components: list[str] | None = None,
) -> UpdateWeightTpuSync:
    args = Namespace(
        ft_components=ft_components or [],
        update_weight_engine_request_timeout=10.0,
    )
    protocol = UpdateWeightTpuSync(args)
    protocol.connect(
        rollout_engines=engines,
        engine_gpu_counts=tp_sizes,
        engine_gpu_offsets=[0] * len(engines),
        engine_cell_ids=cell_ids or [f"cell-{i}" for i in range(len(engines))],
        parallel_state=MagicMock(),
        placement=MagicMock(),
        selector="all",
    )
    return protocol


class TestProtocolFactoryAndProperties:
    def test_factory_returns_tpu_sync_protocol(self):
        args = Namespace(update_weight_transfer_mode="tpu-sync", colocate=False)
        protocol = get_weight_transfer_protocol(args)
        assert isinstance(protocol, UpdateWeightTpuSync)
        assert isinstance(protocol, WeightTransferProtocol)
        assert protocol.streams_buckets is False
        assert WeightTransferProtocol.streams_buckets is True

    def test_send_bucket_is_rejected(self):
        protocol = UpdateWeightTpuSync(Namespace())
        with pytest.raises(NotImplementedError, match="transfer"):
            protocol.send_bucket([])


class TestTpuSyncMetadataAndInvariants:
    def test_1d_whole_and_2d_dim0_sharded_variable_metadata_protos(self, monkeypatch):
        env = _FakeTpuSyncEnv()
        env.install(monkeypatch)

        norm_1d = torch.randn(16, dtype=torch.bfloat16)
        proj_2d_sharded = torch.randn(8, 16, dtype=torch.bfloat16)
        proj_2d_whole = torch.randn(12, 16, dtype=torch.float32)

        v1d = tpu_sync._trainer_variable(0, "model.norm.weight", norm_1d, None)
        assert v1d == _FakeVariableMetadataProto(
            name="model.norm.weight",
            shape=[16],
            mesh_shape=[1],
            layout=[0],
            item_size=2,
            layer_idx=0,
            sharding_spec=[""],
            global_shard_indices=[0],
        )

        v2d_sharded = tpu_sync._trainer_variable(1, "model.layers.0.self_attn.q_proj.weight", proj_2d_sharded, (4, 2))
        assert v2d_sharded == _FakeVariableMetadataProto(
            name="model.layers.0.self_attn.q_proj.weight",
            shape=[32, 16],
            mesh_shape=[4, 1],
            layout=[1, 0],
            item_size=2,
            layer_idx=1,
            sharding_spec=["fsdp", ""],
            global_shard_indices=[2],
        )

        v2d_whole = tpu_sync._trainer_variable(2, "lm_head.weight", proj_2d_whole, None)
        assert v2d_whole == _FakeVariableMetadataProto(
            name="lm_head.weight",
            shape=[12, 16],
            mesh_shape=[1, 1],
            layout=[1, 0],
            item_size=4,
            layer_idx=2,
            sharding_spec=["", ""],
            global_shard_indices=[0],
        )

    def test_1d_sharded_tensor_is_rejected(self):
        t1d = torch.randn(4, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="must be registered whole"):
            tpu_sync._check_bound_tensor("model.norm.weight", t1d, (4, 0), seen_data_ptrs=set())
        with pytest.raises(ValueError, match="must be registered whole"):
            tpu_sync._trainer_variable(0, "model.norm.weight", t1d, (4, 0))

    def test_iterator_contract_violations_are_rejected_not_patched(self):
        """Views, non-contiguous and aliased buffers are iterator bugs; the protocol refuses them loudly."""
        base = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4).clone()
        cases = {
            "offset view": (base[2:5], "is a view"),
            "zero-offset view": (base[:3], "is a view"),
            "transposed": (base.transpose(0, 1), "not contiguous"),
        }
        for label, (tensor, message) in cases.items():
            with pytest.raises(ValueError, match=message):
                tpu_sync._check_bound_tensor(label, tensor, None, seen_data_ptrs=set())

        seen: set[int] = set()
        tpu_sync._check_bound_tensor("model.embed_tokens.weight", base, None, seen_data_ptrs=seen)
        with pytest.raises(ValueError, match="aliases the storage"):
            tpu_sync._check_bound_tensor("lm_head.weight", base, None, seen_data_ptrs=seen)

        owning = torch.ones(4, 4, dtype=torch.bfloat16)
        tpu_sync._check_bound_tensor("w", owning, (2, 1), seen_data_ptrs=set())

    def test_a_view_on_one_rank_fails_every_rank_before_the_controller_starts(self, monkeypatch):
        env = _FakeTpuSyncEnv()
        env.install(monkeypatch)
        _patch_dist(monkeypatch, rank=0, world_size=2, other_rank_errors=[None])
        protocol = _make_protocol([_RecordingRolloutClient(env=env, tp_size=1, events=[], engine_label="e0")], [1])
        base = torch.ones(8, 8, dtype=torch.bfloat16)

        with pytest.raises(RuntimeError, match=r"check local weights failed on rank 0: ValueError: .*is a view"):
            protocol.transfer(SimpleNamespace(iter_local_weights=lambda: [("w", base[2:6], (2, 0))]))

        assert env.controllers == [] and env.synchronizers == []


class TestTpuSyncTransferFlow:
    def test_controller_startup_broadcast_and_non_zero_rank_registration(self, monkeypatch):
        """Non-zero ranks receive the broadcast controller address without starting a server or calling engines."""
        env = _FakeTpuSyncEnv()
        env.install(monkeypatch)
        dist_state = _patch_dist(
            monkeypatch, rank=2, world_size=4, broadcast_address="10.4.17.6:41234", node_ip="10.4.17.7"
        )

        events: list = []
        engine = _RecordingRolloutClient(env=env, tp_size=4, events=events, engine_label="e0")
        protocol = _make_protocol([engine], [4])
        iterator = SimpleNamespace(
            iter_local_weights=lambda: [
                ("model.norm.weight", torch.ones(8, dtype=torch.bfloat16), None),
                ("model.layers.0.mlp.weight", torch.ones(4, 8, dtype=torch.bfloat16), (4, 2)),
            ]
        )

        protocol.transfer(iterator)

        assert env.controllers == []
        assert env.servers == []
        assert dist_state["broadcasts"] == ["10.4.17.6:41234"]
        assert events == []
        assert len(env.registrations) == 1
        addr, unit_id, reg = env.registrations[0]
        assert addr == "10.4.17.6:41234"
        assert unit_id == _FakeRaidenId(job_name="trainer", job_replica_id="2", data_name="weights")
        assert reg["mesh_shape"] == [4, 1]
        assert reg["mesh_axes"] == ["fsdp", "tp"]
        assert reg["shards"] == ["10.4.17.7:50000"]
        assert reg["control_plane_rpc_address"] == "10.4.17.7:50001"
        assert env.synchronizers[0].unbind_count == 1
        assert env.synchronizers[0].d2h_count == 0
        protocol.close()

    def test_multi_engine_init_single_transfer_async_wait_install_and_step2_reuse(self, monkeypatch):
        """Multiple engines get distinct sampler0/sampler1 names, one start_transfer, async wait, and step-2 reuse."""
        env = _FakeTpuSyncEnv(server_port=45678)
        env.install(monkeypatch)
        dist_state = _patch_dist(monkeypatch, rank=0, world_size=1, node_ip="10.4.17.6")

        events: list[tuple[str, str, dict]] = []
        engine0 = _RecordingRolloutClient(env=env, tp_size=4, events=events, engine_label="e0")
        engine1 = _RecordingRolloutClient(env=env, tp_size=2, events=events, engine_label="e1")
        protocol = _make_protocol([engine0, engine1], [4, 2], ["cell-a", "cell-b"])

        base = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
        step1_weights = [
            ("model.norm.weight", torch.ones(8, dtype=torch.bfloat16), None),
            ("model.layers.0.self_attn.q_proj.weight", base[2:6].clone(), (2, 0)),
        ]
        step2_weights = [
            ("model.norm.weight", torch.full((8,), 2.0, dtype=torch.bfloat16), None),
            ("model.layers.0.self_attn.q_proj.weight", torch.full((4, 8), 3.0, dtype=torch.bfloat16), (2, 0)),
        ]

        # Step 1: starts controller, binds + registers, inits both receivers, runs 1 transfer, installs, unbinds.
        protocol.transfer(SimpleNamespace(iter_local_weights=lambda: list(step1_weights)))

        assert len(env.controllers) == 1
        assert len(env.servers) == 1 and env.servers[0].started and not env.servers[0].stopped
        assert dist_state["broadcasts"] == ["10.4.17.6:45678"]
        assert len(env.synchronizers) == 1
        ws = env.synchronizers[0]
        assert ws.init_kwargs == {
            "local_port": 0,
            "parallelism": 8,
            "listener_port": 0,
            "bind_ip": "10.4.17.6",
            "unsafe_skip_buffer_lock": True,
            "auto_h2d": False,
        }
        assert ws.d2h_count == 0
        assert ws.unbind_count == 1

        # Both engines got init_tpu_sync_receiver with distinct sampler0 / sampler1 job names before install.
        init_events = [e for e in events if e[1] == "init_tpu_sync_receiver"]
        install_events = [e for e in events if e[1] == "install_tpu_sync_weights"]
        assert len(init_events) == 2
        assert len(install_events) == 2
        assert {e[2]["job_name"] for e in init_events} == {"sampler0", "sampler1"}
        for _, _, payload in init_events:
            assert payload["controller_address"] == "10.4.17.6:45678"
            assert payload["weight_shapes"] == {
                "model.norm.weight": [8],
                "model.layers.0.self_attn.q_proj.weight": [8, 8],
            }
            assert payload["parallelism"] == 8

        # Single start_transfer covers trainer + all TP ranks of sampler0 (4) and sampler1 (2), awaited asynchronously.
        ctrl = env.controllers[0]
        assert len(ctrl.transfers) == 1
        kwargs, fut1 = ctrl.transfers[0]
        assert kwargs["src_units"] == [_FakeRaidenId("trainer", "0", "weights")]
        assert kwargs["dst_units"] == [
            _FakeRaidenId("sampler0", "0", "weights"),
            _FakeRaidenId("sampler0", "1", "weights"),
            _FakeRaidenId("sampler0", "2", "weights"),
            _FakeRaidenId("sampler0", "3", "weights"),
            _FakeRaidenId("sampler1", "0", "weights"),
            _FakeRaidenId("sampler1", "1", "weights"),
        ]
        assert kwargs["dst_mem_type"] == _FakeRaidenMemoryType.DRAM
        assert kwargs["use_block_chunks"] is True
        assert kwargs["is_sender"] is True
        assert kwargs["expected_block_count"] == 0
        assert kwargs["parallelism"] == 8
        assert fut1.awaited is True
        assert fut1.wait_threadsafe_called is False

        # Step 2: reuses controller, reuses WeightSynchronizer via bind_weights, skips init_tpu_sync_receiver.
        events.clear()
        protocol.transfer(SimpleNamespace(iter_local_weights=lambda: list(step2_weights)))

        assert len(env.controllers) == 1
        assert len(env.synchronizers) == 1
        assert len(env.registrations) == 1
        assert len(ws.bound_batches) == 2
        assert ws.unbind_count == 2
        assert [e[1] for e in events] == ["install_tpu_sync_weights", "install_tpu_sync_weights"]
        assert len(ctrl.transfers) == 2
        assert ctrl.transfers[1][1].awaited is True

        # close() shuts down synchronizer and stops controller server cleanly.
        protocol.close()
        assert ws.shutdown_count == 1
        assert env.servers[0].stopped is True

    def test_fallback_shutdown_and_recreate_when_unbind_weights_is_absent(self, monkeypatch):
        env = _FakeTpuSyncEnv(sync_cls=_FakeWeightSynchronizerNoUnbind)
        env.install(monkeypatch)
        _patch_dist(monkeypatch, rank=0, world_size=1)

        events: list = []
        engine = _RecordingRolloutClient(env=env, tp_size=2, events=events, engine_label="e0")
        protocol = _make_protocol([engine], [2])
        weights = [("model.norm.weight", torch.ones(8, dtype=torch.bfloat16), None)]

        protocol.transfer(SimpleNamespace(iter_local_weights=lambda: list(weights)))
        assert len(env.synchronizers) == 1
        assert env.synchronizers[0].shutdown_count == 1

        # Step 2 cleanly recreates WeightSynchronizer and re-registers trainer without re-initializing engines.
        events.clear()
        protocol.transfer(SimpleNamespace(iter_local_weights=lambda: list(weights)))
        assert len(env.synchronizers) == 2
        assert env.synchronizers[1].shutdown_count == 1
        assert len(env.registrations) == 2
        assert [e[1] for e in events] == ["install_tpu_sync_weights"]
        protocol.close()

    def test_controller_server_and_synchronizer_cleaned_up_on_transfer_error(self, monkeypatch):
        env = _FakeTpuSyncEnv()
        env.install(monkeypatch)
        _patch_dist(monkeypatch, rank=0, world_size=1)

        events: list = []
        engine = _RecordingRolloutClient(env=env, tp_size=1, events=events, engine_label="e0")
        protocol = _make_protocol([engine], [1])

        orig_make = sys.modules["tpu_sync.rpc.raiden_controller"].RaidenController

        def failing_controller(port: int = 0):
            ctrl = orig_make(port=port)
            ctrl.next_transfer_error = RuntimeError("tcp reset")
            return ctrl

        monkeypatch.setattr(sys.modules["tpu_sync.rpc.raiden_controller"], "RaidenController", failing_controller)

        weights = [("model.norm.weight", torch.ones(8, dtype=torch.bfloat16), None)]
        with pytest.raises(RuntimeError, match="tcp reset"):
            protocol.transfer(SimpleNamespace(iter_local_weights=lambda: list(weights)))

        assert env.synchronizers[0].unbind_count == 1
        assert env.synchronizers[0].shutdown_count == 1
        assert env.servers[0].stopped is True
        assert protocol._controller_address is None
        assert protocol._synchronizer is None

    def test_controller_server_stopped_if_start_raises(self, monkeypatch):
        env = _FakeTpuSyncEnv(fail_server_start=True)
        env.install(monkeypatch)
        _patch_dist(monkeypatch, rank=0, world_size=1)

        engine = _RecordingRolloutClient(env=env, tp_size=1, events=[], engine_label="e0")
        protocol = _make_protocol([engine], [1])
        weights = [("model.norm.weight", torch.ones(8, dtype=torch.bfloat16), None)]

        with pytest.raises(RuntimeError, match="bind failed"):
            protocol.transfer(SimpleNamespace(iter_local_weights=lambda: list(weights)))

        assert len(env.servers) == 1
        assert env.servers[0].stopped is True

    def test_engine_agnostic_rpcs_work_with_real_vllm_and_sglang_clients(self, monkeypatch):
        """Real VllmApiClient and SGLangApiClient both satisfy UpdateWeightTpuSync's engine calls."""
        from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
        from miles.backends.vllm_utils.vllm_api_client import VllmApiClient

        env = _FakeTpuSyncEnv(server_port=49100)
        env.install(monkeypatch)
        _patch_dist(monkeypatch, rank=0, world_size=1, node_ip="10.4.17.6")

        vllm_posts: list[tuple[str, dict]] = []
        sglang_reqs: list[tuple[str, dict]] = []

        @dataclasses.dataclass(frozen=True)
        class _RecordingVllmClient(VllmApiClient):
            async def _post(self, endpoint, payload=None, *, params=None, timeout=None):
                vllm_posts.append((endpoint, payload))
                if endpoint == "collective_rpc" and payload["method"] == "init_tpu_sync_on_worker":
                    job = payload["kwargs"]["job_name"]
                    ctrl = env.controllers[-1]
                    for r in range(2):
                        ctrl.registered_units[_FakeRaidenId(job, str(r), "weights")] = {}
                return {"results": [{"ok": True}]}

        @dataclasses.dataclass(frozen=True)
        class _RecordingSglangClient(SGLangApiClient):
            async def _make_request(self, endpoint, payload=None, *, timeout=None):
                sglang_reqs.append((endpoint, payload))
                if endpoint == "plugin_weight_update" and payload["handler"] == "tpu_sync.init_receiver":
                    job = payload["payload"]["job_name"]
                    ctrl = env.controllers[-1]
                    for r in range(2):
                        ctrl.registered_units[_FakeRaidenId(job, str(r), "weights")] = {}
                return {"success": True}

        protocol = _make_protocol(
            [_RecordingVllmClient("http://vllm:8000"), _RecordingSglangClient("http://sglang:30000")],
            [2, 2],
            ["vllm-cell-0", "sglang-cell-1"],
        )
        weights = [("model.norm.weight", torch.ones(8, dtype=torch.bfloat16), None)]
        protocol.transfer(SimpleNamespace(iter_local_weights=lambda: list(weights)))
        protocol.close()

        assert vllm_posts == [
            (
                "collective_rpc",
                {
                    "method": "init_tpu_sync_on_worker",
                    "args": [],
                    "kwargs": {
                        "controller_address": "10.4.17.6:49100",
                        "job_name": "sampler0",
                        "weight_shapes": {"model.norm.weight": [8]},
                        "parallelism": 8,
                    },
                    "timeout": None,
                },
            ),
            (
                "collective_rpc",
                {"method": "install_tpu_sync_weights", "args": [], "kwargs": {}, "timeout": None},
            ),
        ]
        assert sglang_reqs == [
            (
                "plugin_weight_update",
                {
                    "handler": "tpu_sync.init_receiver",
                    "payload": {
                        "controller_address": "10.4.17.6:49100",
                        "job_name": "sampler1",
                        "weight_shapes": {"model.norm.weight": [8]},
                        "parallelism": 8,
                    },
                    "flush_cache": False,
                },
            ),
            (
                "plugin_weight_update",
                {"handler": "tpu_sync.install", "payload": {}, "flush_cache": False},
            ),
        ]


class TestWeightUpdaterWholeModelTransfer:
    def test_updater_calls_transfer_without_streaming_buckets(self, monkeypatch):
        """With streams_buckets=False, WeightUpdater runs transfer() inside the session frame and streams no buckets."""
        calls: list[tuple[str, dict]] = []

        class _FakeClient:
            def __getattr__(self, name: str):
                async def _rpc(**kwargs):
                    calls.append((name, kwargs))
                    return {"success": True}

                return _rpc

        transferred: list[object] = []

        class _WholeModelProtocol(WeightTransferProtocol):
            streams_buckets = False

            def connect(
                self,
                rollout_engines,
                engine_gpu_counts,
                engine_gpu_offsets,
                engine_cell_ids,
                parallel_state,
                placement,
                selector,
            ) -> None:
                self.rollout_engines = rollout_engines
                self.is_sender = True
                self.selector_seen = selector
                from miles.backends.training_utils.weight_update.rollout_cell_updater import (
                    create_rollout_cell_updaters,
                )

                self.cell_updaters_of_cell_id = create_rollout_cell_updaters(
                    self.args, rollout_engines, engine_cell_ids
                )

            def send_bucket(self, bucket):
                raise AssertionError("send_bucket must not be called when streams_buckets=False")

            def transfer(self, hf_weight_iterator) -> None:
                transferred.append(hf_weight_iterator)

            def after_base_weights(self) -> None:
                calls.append(("after_base_weights", {}))

        args = Namespace(
            update_weight_transfer_mode="tpu-sync",
            colocate=False,
            fully_async=False,
            pause_generation_mode="retract",
            check_lora_weight_equal=False,
            enable_mtp_training=False,
            ft_components=[],
            update_weight_engine_request_timeout=10.0,
            # Mirrors the fallback base_args registered when sglang is not installed (--rollout-engine vllm).
            sglang=SimpleNamespace(base_args={"speculative_algorithm": None}),
        )
        protocol = _WholeModelProtocol(args)
        iterator = MagicMock()
        iterator.placement = MagicMock(gather_pp=False)

        with (
            patch.object(updater_mod, "get_weight_transfer_protocol", return_value=protocol),
            patch.object(updater_mod, "dist") as dist_mock,
            patch.object(updater_mod, "get_gloo_group", return_value="gloo"),
        ):
            dist_mock.get_rank.return_value = 0
            dist_mock.get_world_size.return_value = 1
            dist_mock.all_gather_object.side_effect = lambda out, val, group: out.__setitem__(0, val)

            updater = WeightUpdater(
                args,
                [MagicMock()],
                weights_getter=lambda: {},
                model_name="qwen3",
                quantization_config=None,
                iterator_factory=lambda *a, **k: iterator,
                parallel_state=MagicMock(),
                is_lora=False,
            )
            updater.connect_rollout_engines([_FakeClient()], [4], [0], ["cell-0"])
            assert protocol.selector_seen == "all"
            assert updater.update_weights(weight_version=3) is True

        assert transferred == [iterator]
        iterator.iter_hf_weights.assert_not_called()
        assert [name for name, _ in calls] == [
            "pause_generation",
            "flush_cache",
            "begin_weight_update",
            "after_base_weights",
            "end_weight_update",
            "update_weight_version",
            "continue_generation",
        ]


class TestRankErrors:
    def test_remote_rank_failure_raises_on_passing_rank(self, monkeypatch):
        _patch_dist(monkeypatch, rank=0, world_size=3, other_rank_errors=[None, "OSError: disk full"])
        with pytest.raises(RuntimeError, match="bind failed on rank 2: OSError: disk full"):
            with fail_on_every_rank("bind"):
                pass

    def test_local_failure_preserves_cause(self, monkeypatch):
        _patch_dist(monkeypatch, rank=0, world_size=2, other_rank_errors=[None])
        with pytest.raises(RuntimeError, match="failed on rank 0: ValueError: bad shard") as exc_info:
            with fail_on_every_rank("bind"):
                raise ValueError("bad shard")
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_all_passing_ranks_succeed(self, monkeypatch):
        _patch_dist(monkeypatch, rank=1, world_size=3, other_rank_errors=[None, None])
        with fail_on_every_rank("bind"):
            pass


class TestTitanLocalWeightsExport:
    def test_iter_local_weights_exports_dim0_shards_and_gathers_1d_using_parallelism(self, monkeypatch, tmp_path):
        """Uses main's trainer.parallelism (not parallel_dims), keeps even 2D Shard(0) local, gathers 1D whole."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        store = FakeStore()
        torch.distributed.init_process_group(backend="fake", rank=1, world_size=4, store=store)
        try:
            mesh_1d = init_device_mesh("cpu", (4,), mesh_dim_names=("fsdp",))
            mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))

            full_2d = torch.arange(64, dtype=torch.float32).reshape(8, 8)
            local_shard_2d = full_2d[2:4].clone()
            dt_2d_even = DTensor.from_local(local_shard_2d, mesh_1d, (Shard(0),), shape=(8, 8), stride=(8, 1))

            full_1d = torch.arange(8, dtype=torch.float32)
            local_shard_1d = full_1d[2:4].clone()
            dt_1d = DTensor.from_local(local_shard_1d, mesh_1d, (Shard(0),), shape=(8,), stride=(1,))

            # 2D uneven dim-0 shard (5 not divisible by 4) -> must gather whole.
            dt_2d_uneven = DTensor.from_local(torch.ones(1, 4), mesh_1d, (Shard(0),), shape=(5, 4), stride=(4, 1))
            # 2D mesh -> must gather whole.
            dt_2d_mesh = DTensor.from_local(
                torch.ones(2, 4), mesh_2d, (Shard(0), Replicate()), shape=(4, 4), stride=(4, 1)
            )
            # Plain view tensor -> must clone into owning storage.
            view_base = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            plain_view = view_base[:2]

            assert fsdp_dim0_block(dt_2d_even) == (4, 1)
            assert fsdp_dim0_block(dt_1d) is None
            assert fsdp_dim0_block(dt_2d_uneven) is None
            assert fsdp_dim0_block(dt_2d_mesh) is None
            assert fsdp_dim0_block(plain_view) is None

            state = {
                "norm.weight": dt_1d,
                "proj.weight": dt_2d_even,
                "view.weight": plain_view,
                "layers.0.attn.wq.lora_a.weight": torch.zeros(2, 2),
            }
            part = SimpleNamespace(state_dict=lambda: state)
            adapter = SimpleNamespace(to_hf=lambda d: {f"hf.{k}": v for k, v in d.items()})
            trainer = SimpleNamespace(
                parallelism=SimpleNamespace(get_optional_mesh=lambda name: None),
                model_parts=[part],
                engine=SimpleNamespace(state_dict_adapter=adapter),
            )

            monkeypatch.setattr(titan_iter_mod, "load_hf_config", lambda path: SimpleNamespace())
            monkeypatch.setattr(titan_iter_mod.current_platform, "current_device", lambda: torch.device("cpu"))
            monkeypatch.setattr(
                titan_iter_mod,
                "_checkpoint_dtypes",
                lambda path: {"hf.norm.weight": torch.bfloat16, "hf.proj.weight": torch.bfloat16},
            )
            monkeypatch.setattr(
                titan_iter_mod,
                "gather_full_param",
                lambda t: full_1d.clone() if t is dt_1d else t,
            )

            it = TitanHfWeightIterator(
                Namespace(hf_checkpoint=str(tmp_path), update_weight_buffer_size=1024),
                trainer,
                placement=MagicMock(gather_pp=False),
                model_name="qwen3",
                quantization_config=None,
            )
            exported = {name: (tensor, block) for name, tensor, block in it.iter_local_weights()}

            assert set(exported.keys()) == {"hf.norm.weight", "hf.proj.weight", "hf.view.weight"}
            norm_t, norm_block = exported["hf.norm.weight"]
            assert norm_block is None
            assert norm_t.shape == (8,) and norm_t.dtype == torch.bfloat16
            assert norm_t._base is None and norm_t.storage_offset() == 0

            proj_t, proj_block = exported["hf.proj.weight"]
            assert proj_block == (4, 1)
            assert proj_t.shape == (2, 8) and proj_t.dtype == torch.bfloat16
            assert torch.equal(proj_t, local_shard_2d.to(torch.bfloat16))
            assert proj_t._base is None and proj_t.storage_offset() == 0

            view_t, view_block = exported["hf.view.weight"]
            assert view_block is None
            assert view_t._base is None and view_t.storage_offset() == 0
            assert view_t.untyped_storage().nbytes() == view_t.numel() * view_t.element_size()

            # Pipeline parallelism is explicitly rejected via trainer.parallelism.
            trainer_pp = SimpleNamespace(
                parallelism=SimpleNamespace(get_optional_mesh=lambda name: "pp_mesh" if name == "pp" else None),
                model_parts=[part],
                engine=SimpleNamespace(state_dict_adapter=adapter),
            )
            it.model = trainer_pp
            with pytest.raises(NotImplementedError, match="pipeline parallelism"):
                list(it.iter_local_weights())

            # Grouped MoE experts are explicitly rejected.
            moe_part = SimpleNamespace(state_dict=lambda: {"layers.0.routed_experts.w1": torch.ones(4, 2, 2)})
            trainer_moe = SimpleNamespace(
                parallelism=SimpleNamespace(get_optional_mesh=lambda name: None),
                model_parts=[moe_part],
                engine=SimpleNamespace(state_dict_adapter=adapter),
            )
            it.model = trainer_moe
            with pytest.raises(NotImplementedError, match="grouped MoE experts"):
                list(it.iter_local_weights())
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
