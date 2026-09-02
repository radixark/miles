import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from tests.fast.ray.rollout.conftest import make_args

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.inference_controller import (
    InferenceController,
    UpdatableEngines,
    _compute_server_cell_meta_from_info,
)
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.ray.specs.inference import compute_engine_pool_ids, compute_router_pool_id, specs_inference_engine
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.registration.models import RegisteredCellInfo, RegistrationSnapshot
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts, WorkerMetaContext

_RUN_UUID = "run-uuid-1"


def _make_cell_info(
    *,
    cell_id: str = "inference-engine-0-0-0",
    workers_hash: str = "pseudo-hash-0",
    alive: bool = True,
    model_id: str = "model-a",
    pool_id: str = "inference-engine-0-0",
) -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        pool_id=pool_id,
        alive=alive,
        worker_names=[f"{cell_id}-0"],
        workers_hash=workers_hash,
        meta=dict(
            model_id=model_id,
            worker_type="regular",
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            needs_offload=False,
            update_weights=True,
        ),
    )


def _make_cell_meta(info: CellInfo) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id=info.meta["model_id"],
        worker_type=info.meta["worker_type"],
        cell_id=info.cell_id,
        num_gpus_per_engine=info.meta["num_gpus_per_engine"],
        gpu_offset=info.meta["gpu_offset"],
        sglang_api_key=info.meta["sglang_api_key"],
        worker_name=info.worker_names[0],
        needs_offload=info.meta["needs_offload"],
        update_weights=info.meta["update_weights"],
        workers_hash=info.workers_hash,
    )


class _RecordingServer:
    def __init__(
        self,
        server_cells: dict | None = None,
        *,
        model_name: str = "model",
        update_weights: bool = False,
        cells_gate: asyncio.Event | None = None,
    ):
        self.server_cells = server_cells or {}
        self.health_checker_activeness = ActivenessTracker(active=True)
        self.update_weights = update_weights
        self.model_name = model_name
        self.calls: list[tuple] = []
        self.router_ip: str = "10.0.0.9"
        self.router_port: int = 31000
        self.api_clients: list = []
        self.engine_gpu_counts: list[int] = []
        self.engine_gpu_offsets: list[int] = []
        self.offload_tags: list = []
        self.onload_tags: list = []
        self.check_weights_kwargs: list[dict] = []
        self.waited_init_expected_num_cells = 0
        self.cells_timeouts: list[float] = []
        self.dispose_count = 0
        self.faulted_cells: list[str] = []
        self._cells_gate = cells_gate

    def addressable_cell_ids(self) -> list[str]:
        return [cell_id for cell_id in self.server_cells if cell_id not in self.faulted_cells]

    def mark_cell_faulted(self, cell_id: str) -> None:
        self.faulted_cells.append(cell_id)

    async def offload(self, tags=None):
        self.calls.append(("offload",))
        self.offload_tags.append(tags)

    async def onload(self, tags=None):
        self.onload_tags.append(tags)

    async def dispose(self):
        self.dispose_count += 1

    async def check_weights(self, action, allow_quant_error=False, selector="all", skip_list=None):
        self.calls.append(("check_weights", action))
        self.check_weights_kwargs.append(
            dict(action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list)
        )
        return [self.model_name]

    async def add_cell(self, cell_meta: ServerCellMetadata):
        self.calls.append(("add", cell_meta.cell_id))
        self.server_cells[cell_meta.cell_id] = SimpleNamespace(meta=cell_meta)

    async def remove_cell(self, cell_id: str):
        self.calls.append(("remove", cell_id))
        del self.server_cells[cell_id]

    async def wait_init_expected_num_cells(self, timeout: float = 3600.0) -> None:
        if self._cells_gate is not None:
            await self._cells_gate.wait()
        self.waited_init_expected_num_cells += 1
        self.cells_timeouts.append(timeout)


class _RecordingRemoteMethod:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def remote(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _FakeUpdatableCell:
    def __init__(self, workers_hash: str):
        self.meta = SimpleNamespace(workers_hash=workers_hash)
        self.marked_ready = 0
        self.is_pending_weights = True
        self.is_pending_weights_or_serving = True

    async def mark_weights_ready(self) -> None:
        self.marked_ready += 1


class _TickingCell:
    def __init__(self, cell_id: str = "engine-0"):
        self.meta = SimpleNamespace(cell_id=cell_id)
        self.tick_count = 0

    async def tick(self) -> None:
        self.tick_count += 1


class _RecordingEvalFleet:
    def __init__(self, args: Namespace, *, srv):
        self.args = args
        self.srv = srv

    async def dispose(self) -> None:
        return None


class _FakeWorkerProvider(BaseWorkerProvider):
    def __init__(self, cell_infos: list[CellInfo], *, pool_ids: list[str] | None = None) -> None:
        self._cell_infos = cell_infos
        self._pools = pool_ids or []
        self._worker_manager_handle = SimpleNamespace(inject_fault=_RecordingRemoteMethod())
        self.watched_pool_ids: list[str] | None = None
        self.initialized = False
        self.stop_watch_calls = 0

    async def init(self) -> None:
        self.initialized = True

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"the controller must not ask this fake for {worker_name}")

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [[] for _ in cell_ids]

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        assert self.initialized, "the controller must init the provider before observing its cells"
        self.watched_pool_ids = list(self._pools)
        for info in self._cell_infos:
            if info.pool_id in self._pools:
                await reconcile(info.cell_id, info)

        async def _stop_watch() -> None:
            self.stop_watch_calls += 1

        return _stop_watch


class _RecordingInferenceControllerEvalFleet:
    def __init__(self, info: EvalFleetInfo):
        self.info = info
        self.pins: list[dict] = []

    async def pin(self, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        self.pins.append(dict(checkpoint_dir=checkpoint_dir, weight_version=weight_version))
        return EvalFleetPin(skip_reason=None)


def _make_controller(
    servers: dict,
    *,
    engine_provider: BaseWorkerProvider | None = None,
    registration_hub: RegistrationHub | None = None,
) -> InferenceController:
    engines = registration_hub if registration_hub is not None else engine_provider
    engines = engines if engines is not None else _FakeWorkerProvider([])

    controller = InferenceController.__new__(InferenceController)
    controller.args = SimpleNamespace(
        debug_train_only=False, use_fault_tolerance=False, ci_test=False, colocate=False, run_uuid=_RUN_UUID
    )
    controller.servers = servers
    controller.context_lock = ContextLock("InferenceController")
    controller._engine_provider = engines
    controller._router_providers = [_FakeWorkerProvider([])]
    return controller


class TestHealthCheckerActiveness:
    @pytest.mark.asyncio
    async def test_offload_pauses_probing_before_putting_engines_to_sleep(self):
        """A slept engine cannot answer /health_generate, so probing must stop first."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})

        await controller.offload()

        assert not srv.health_checker_activeness.get().active
        assert srv.calls == [("offload",)]

    @pytest.mark.asyncio
    async def test_starting_a_weight_update_pauses_probing(self):
        """Engines are unusable while their weights are being replaced."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})

        info = await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        assert not srv.health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_preparing_a_rollout_resumes_probing(self):
        """Probing comes back exactly when the engines start serving traffic again."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})
        srv.health_checker_activeness.bump_active(False)

        await controller.prepare_rollout(rollout_id=0)

        assert srv.health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_preparing_a_rollout_awaits_the_dashboard_engine_registration(self, monkeypatch):
        """The dashboard hook is a coroutine, so prepare_rollout must await it instead of leaving it unscheduled."""
        awaited: list[tuple[dict, _FakeWorkerProvider]] = []

        async def _record(servers: dict, *, provider: _FakeWorkerProvider) -> None:
            awaited.append((servers, provider))

        monkeypatch.setattr(dashboard_hooks, "register_engines", _record)
        servers = {"default": _RecordingServer()}
        engine_provider = _FakeWorkerProvider([])
        controller = _make_controller(servers, engine_provider=engine_provider)

        await controller.prepare_rollout(rollout_id=0)

        assert awaited == [(servers, engine_provider)]

    @pytest.mark.asyncio
    async def test_preparing_an_eval_resumes_probing(self):
        """Eval drives the same engines as a rollout does."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})
        srv.health_checker_activeness.bump_active(False)

        await controller.prepare_eval()

        assert srv.health_checker_activeness.get().active


class TestRolloutFaultInjectionWindow:
    @pytest.mark.asyncio
    async def test_fault_injection_reaches_a_serving_rollout_cell(self) -> None:
        """A serving rollout cell accepts a fault through the Ray worker manager."""
        cell_id = "inference-engine-0-0-0"
        server = _RecordingServer(server_cells={cell_id: object(), "inference-engine-0-1-0": object()})
        provider = _FakeWorkerProvider([])
        controller = _make_controller({"default": server}, engine_provider=provider)

        await controller.inject_fault_between_weight_updates(
            cell_id=cell_id,
            mode=FailureMode.SIGKILL,
            sub_index=0,
        )

        assert provider._worker_manager_handle.inject_fault.calls == [
            ((cell_id,), {"mode": "sigkill", "worker_in_cell_index": 0, "wait_until_applied": True})
        ]

    @pytest.mark.asyncio
    async def test_an_injected_cell_stops_being_addressable_before_the_lock_is_released(self) -> None:
        """The next weight update snapshots the engines under this same lock, and must not snapshot a corpse."""
        cell_id = "inference-engine-0-0-0"
        server = _RecordingServer(server_cells={cell_id: object(), "inference-engine-0-1-0": object()})
        controller = _make_controller({"default": server}, engine_provider=_FakeWorkerProvider([]))

        await controller.inject_fault_between_weight_updates(
            cell_id=cell_id,
            mode=FailureMode.SIGKILL,
            sub_index=0,
        )

        assert server.faulted_cells == [cell_id]

    @pytest.mark.asyncio
    async def test_fault_injection_refuses_the_last_addressable_rollout_cell(self) -> None:
        """Killing the only engine that can serve leaves the next rollout with nowhere to go."""
        cell_id = "inference-engine-0-0-0"
        server = _RecordingServer(server_cells={cell_id: object()})
        provider = _FakeWorkerProvider([])
        controller = _make_controller({"default": server}, engine_provider=provider)

        with pytest.raises(RuntimeError, match="refusing fault injection because nothing would serve"):
            await controller.inject_fault_between_weight_updates(
                cell_id=cell_id,
                mode=FailureMode.SIGKILL,
                sub_index=0,
            )

        assert provider._worker_manager_handle.inject_fault.calls == []

    @pytest.mark.asyncio
    async def test_fault_injection_ignores_an_already_faulted_cell_when_counting_replicas(self) -> None:
        """A cell killed a moment ago still sits in server_cells until reconcile removes it."""
        cell_id = "inference-engine-0-0-0"
        server = _RecordingServer(server_cells={cell_id: object(), "inference-engine-0-1-0": object()})
        server.mark_cell_faulted("inference-engine-0-1-0")
        provider = _FakeWorkerProvider([])
        controller = _make_controller({"default": server}, engine_provider=provider)

        with pytest.raises(RuntimeError, match="refusing fault injection because nothing would serve"):
            await controller.inject_fault_between_weight_updates(
                cell_id=cell_id,
                mode=FailureMode.SIGKILL,
                sub_index=0,
            )

        assert provider._worker_manager_handle.inject_fault.calls == []

    @pytest.mark.asyncio
    async def test_fault_injection_refuses_an_offloaded_rollout_cell(self) -> None:
        """Colocate must not kill rollout processes while trainer ranks own the shared GPUs."""
        cell_id = "inference-engine-0-0-0"
        server = _RecordingServer(server_cells={cell_id: object()})
        server.health_checker_activeness.bump_active(False)
        provider = _FakeWorkerProvider([])
        controller = _make_controller({"default": server}, engine_provider=provider)

        with pytest.raises(RuntimeError, match="is offloaded; refusing fault injection"):
            await controller.inject_fault_between_weight_updates(
                cell_id=cell_id,
                mode=FailureMode.SIGKILL,
                sub_index=0,
            )

        assert provider._worker_manager_handle.inject_fault.calls == []


class TestReconcile:
    @pytest.fixture
    def servers(self) -> dict[str, _RecordingServer]:
        return {"model-a": _RecordingServer(), "model-b": _RecordingServer()}

    @pytest.mark.asyncio
    async def test_an_observed_untracked_cell_is_added_to_its_model_server(self, servers):
        """A newly observed engine cell lands in the server named by its model_id meta."""
        controller = _make_controller(servers)
        info = _make_cell_info()

        await controller._reconcile(info.cell_id, info)

        assert servers["model-a"].calls == [("add", info.cell_id)]

    @pytest.mark.asyncio
    async def test_a_second_models_cell_is_routed_to_that_models_server(self, servers):
        """Routing is by model_id, so model-b's cell must not be absorbed by the first server."""
        controller = _make_controller(servers)
        info = _make_cell_info(cell_id="inference-engine-1-0-0", model_id="model-b", pool_id="inference-engine-1-0")

        await controller._reconcile(info.cell_id, info)

        assert servers["model-a"].calls == []
        assert servers["model-b"].calls == [("add", info.cell_id)]

    @pytest.mark.asyncio
    async def test_a_disappeared_tracked_cell_is_removed(self, servers):
        """A tracked cell reported as gone is removed even though no meta is observable."""
        info = _make_cell_info()
        servers["model-a"].server_cells[info.cell_id] = SimpleNamespace(meta=_make_cell_meta(info))
        controller = _make_controller(servers)

        await controller._reconcile(info.cell_id, None)

        assert servers["model-a"].calls == [("remove", info.cell_id)]
        assert servers["model-a"].server_cells == {}

    @pytest.mark.asyncio
    async def test_a_disappeared_cell_is_removed_from_its_owning_server(self, servers):
        """The owner scan must find the server that actually tracks the cell, not the first one."""
        info = _make_cell_info(cell_id="inference-engine-1-0-0", model_id="model-b", pool_id="inference-engine-1-0")
        servers["model-b"].server_cells[info.cell_id] = SimpleNamespace(meta=_make_cell_meta(info))
        controller = _make_controller(servers)

        await controller._reconcile(info.cell_id, None)

        assert servers["model-a"].calls == []
        assert servers["model-b"].calls == [("remove", info.cell_id)]
        assert servers["model-b"].server_cells == {}

    @pytest.mark.asyncio
    async def test_a_workers_hash_change_replaces_the_cell(self, servers):
        """A relaunched cell (new workers_hash) is removed then re-added, in that order."""
        old_info = _make_cell_info(workers_hash="pseudo-hash-0")
        servers["model-a"].server_cells[old_info.cell_id] = SimpleNamespace(meta=_make_cell_meta(old_info))
        controller = _make_controller(servers)
        new_info = _make_cell_info(workers_hash="pseudo-hash-1")

        await controller._reconcile(new_info.cell_id, new_info)

        assert servers["model-a"].calls == [("remove", new_info.cell_id), ("add", new_info.cell_id)]
        assert servers["model-b"].calls == []

    @pytest.mark.asyncio
    async def test_an_unchanged_tracked_cell_is_a_noop(self, servers):
        """A tracked cell observed with the same workers_hash triggers no bookkeeping change."""
        info = _make_cell_info()
        servers["model-a"].server_cells[info.cell_id] = SimpleNamespace(meta=_make_cell_meta(info))
        controller = _make_controller(servers)

        await controller._reconcile(info.cell_id, info)

        assert servers["model-a"].calls == []

    @pytest.mark.asyncio
    async def test_a_disappeared_untracked_cell_is_a_noop(self, servers):
        """A vanished cell that was never tracked (e.g. a router) triggers nothing."""
        controller = _make_controller(servers)

        await controller._reconcile("miles-router-0-0", None)

        assert servers["model-a"].calls == []
        assert servers["model-b"].calls == []


def _patch_init(monkeypatch: pytest.MonkeyPatch, *, servers: dict[str, _RecordingServer]) -> None:
    async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
        return servers

    async def _fake_resolve_router_addrs(args: Namespace, **kwargs: Any) -> dict[str, HostAndPort]:
        return {name: HostAndPort(host="10.0.0.1", port=30000) for name in servers}

    monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
    monkeypatch.setattr(inference_controller_module, "resolve_router_addrs", _fake_resolve_router_addrs)


class _RefusingWorkerProvider(_FakeWorkerProvider):
    """A provider a run must never touch, so touching it is the failure."""

    def __init__(self) -> None:
        super().__init__([])

    async def init(self) -> None:
        raise AssertionError("debug_train_only must not init any worker provider")

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        raise AssertionError("debug_train_only must not watch cells")


async def _init_controller(args: Namespace, *, engine_provider: _FakeWorkerProvider) -> None:
    controller = InferenceController(args, engine_provider=engine_provider, router_providers=[_FakeWorkerProvider([])])
    await controller.init()
    await controller.dispose()


class TestReconcileAfterAFailedInit:
    @pytest.mark.asyncio
    async def test_an_unchanged_observation_rebuilds_a_cell_whose_init_failed(self, monkeypatch):
        """A cell kept after a failed init keeps matching its own observation, so the identical next sweep never retries it."""
        init_calls: list[str] = []

        async def _record_then_raise(cell: ServerCell) -> None:
            init_calls.append(cell.meta.cell_id)
            raise RuntimeError("injected init failure")

        async def _record(cell: ServerCell) -> None:
            init_calls.append(cell.meta.cell_id)

        controller = _make_controller({})
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=controller.context_lock,
            engine_provider=_FakeWorkerProvider([]),
        )
        controller.servers = {"model-a": srv}
        info = _make_cell_info()
        monkeypatch.setattr(ServerCell, "init", _record_then_raise)

        with pytest.raises(RuntimeError, match="injected init failure"):
            await controller._reconcile(info.cell_id, info)

        monkeypatch.setattr(ServerCell, "init", _record)
        await controller._reconcile(info.cell_id, info)

        assert init_calls == [info.cell_id, info.cell_id]
        assert list(srv.server_cells) == [info.cell_id]
        async with controller.context_lock:
            await srv.dispose()

    @pytest.mark.asyncio
    async def test_a_cell_whose_init_failed_is_not_reported_as_a_live_cell(self, monkeypatch):
        """A dropped cell must vanish from the status surface, otherwise the dashboard shows an engine nobody owns."""
        controller = _make_controller({})
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=controller.context_lock,
            engine_provider=_FakeWorkerProvider([]),
        )
        controller.servers = {"model-a": srv}
        info = _make_cell_info()
        monkeypatch.setattr(ServerCell, "init", _raise_async)

        with pytest.raises(RuntimeError, match="injected init failure"):
            await controller._reconcile(info.cell_id, info)

        assert await controller.get_cell_statuses() == {}


class TestPerModelHealthCheckerActiveness:
    @staticmethod
    def _controller(*model_ids: str) -> tuple[InferenceController, dict[str, _RecordingServer]]:
        servers = {model_id: _RecordingServer(model_name=model_id) for model_id in model_ids}
        return _make_controller(servers), servers

    @pytest.mark.asyncio
    async def test_a_weight_update_pauses_only_the_probing_of_its_own_policy(self):
        """The other policy keeps serving through this window, and an unprobed engine is not healed."""
        controller, servers = self._controller("solver", "verifier")
        servers["solver"].update_weights = True
        before = servers["verifier"].health_checker_activeness.get()

        await controller.start_update_weights(model_id="solver")

        assert not servers["solver"].health_checker_activeness.get().active
        assert servers["verifier"].health_checker_activeness.get() == before

    @pytest.mark.asyncio
    async def test_a_rollout_of_one_policy_does_not_resume_the_probing_of_another(self):
        """This is the race: verifier's next round used to un-pause probing of solver's engines
        mid-broadcast, so the checker reported a live cell unhealthy and recycled it."""
        controller, servers = self._controller("solver", "verifier")
        async with controller.context_lock:
            await controller._health_monitoring_pause("solver")

        await controller.prepare_rollout(rollout_id=0, model_id="verifier")

        assert not servers["solver"].health_checker_activeness.get().active
        assert servers["verifier"].health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_a_policy_that_finished_training_leaves_its_engines_probed_again(self):
        """Its last round ends with a weight update, and no later rollout of its own would ever un-pause it."""
        controller, servers = self._controller("solver", "verifier")
        servers["solver"].update_weights = True
        await controller.start_update_weights(model_id="solver")
        await controller.end_update_weights({})

        await controller.prepare_eval(model_id="solver")

        assert servers["solver"].health_checker_activeness.get().active
        assert servers["verifier"].health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_a_rollout_without_a_model_id_resumes_every_policy(self):
        """A single policy run names no model, and must keep resuming the whole fleet."""
        controller, servers = self._controller("solver", "verifier")
        for srv in servers.values():
            srv.health_checker_activeness.bump_active(False)

        await controller.prepare_rollout(rollout_id=0)

        assert all(srv.health_checker_activeness.get().active for srv in servers.values())

    @pytest.mark.asyncio
    async def test_offloading_pauses_every_policy(self):
        """Colocate offloads the whole fleet at once, so every model stops answering probes."""
        controller, servers = self._controller("solver", "verifier")

        await controller.offload()

        assert not any(srv.health_checker_activeness.get().active for srv in servers.values())

    @pytest.mark.asyncio
    async def test_an_eval_resumes_every_policy(self):
        """Eval names no model either; it drives whatever fleet the run deployed."""
        controller, servers = self._controller("solver", "verifier")
        for srv in servers.values():
            srv.health_checker_activeness.bump_active(False)

        await controller.prepare_eval()

        assert all(srv.health_checker_activeness.get().active for srv in servers.values())


class TestInitSubscription:
    @pytest.mark.asyncio
    async def test_init_initializes_the_provider_before_reading_anything_from_it(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A provider that discovers its engines in init() answers an empty fleet until then, so the
        router addresses and the startup barrier would both be sized against nothing."""
        order: list[str] = []

        class _OrderRecordingProvider(_FakeWorkerProvider):
            async def init(self) -> None:
                order.append("init")
                await super().init()

            async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
                order.append("watch_cells")
                return await super().watch_cells(reconcile)

        async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
            order.append("create_rollout_servers")
            return {"default": _RecordingServer()}

        async def _fake_resolve_router_addrs(args: Namespace, **kwargs: Any) -> dict[str, HostAndPort]:
            order.append("resolve_router_addrs")
            return {"default": HostAndPort(host="10.0.0.1", port=30000)}

        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
        monkeypatch.setattr(inference_controller_module, "resolve_router_addrs", _fake_resolve_router_addrs)
        args = make_args()
        provider = _OrderRecordingProvider([], pool_ids=compute_engine_pool_ids(args))

        await _init_controller(args, engine_provider=provider)

        assert order == ["init", "resolve_router_addrs", "create_rollout_servers", "watch_cells"]

    @pytest.mark.asyncio
    async def test_init_watches_the_engine_provider_it_was_handed(self, monkeypatch: pytest.MonkeyPatch):
        """The pools are the provider's own, so the controller may only open a watch on what it was given."""
        args = make_args()
        provider = _FakeWorkerProvider([], pool_ids=compute_engine_pool_ids(args))
        _patch_init(monkeypatch, servers={"default": _RecordingServer()})

        await _init_controller(args, engine_provider=provider)

        assert provider.watched_pool_ids == compute_engine_pool_ids(args)
        assert compute_router_pool_id(0) not in provider.watched_pool_ids
        assert "session-server" not in provider.watched_pool_ids

    @pytest.mark.asyncio
    async def test_init_subscribes_narrowly_enough_to_never_see_a_router_cell(self, monkeypatch: pytest.MonkeyPatch):
        """A router cell carries no engine meta, so reading one as engine meta would kill startup; the
        controller is safe because it subscribes to the engine pools alone."""
        args = make_args()
        assert compute_router_pool_id(0) not in compute_engine_pool_ids(args)

        router_info = CellInfo(
            cell_id="inference-router-0-0",
            pool_id=compute_router_pool_id(0),
            alive=True,
            worker_names=["inference-router-0-0-0"],
            workers_hash="pseudo-hash-router",
            meta={},
        )
        engine_info = _make_cell_info(model_id="default", pool_id=compute_engine_pool_ids(args)[0])
        provider = _FakeWorkerProvider([router_info, engine_info], pool_ids=compute_engine_pool_ids(args))
        srv = _RecordingServer()
        _patch_init(monkeypatch, servers={"default": srv})

        await _init_controller(args, engine_provider=provider)

        assert srv.calls == [("add", engine_info.cell_id)]


class TestEngineMetaContract:
    def test_the_real_spec_meta_roundtrips_into_server_cell_metadata(self, tmp_path: Path):
        """The engine spec's meta dict and the driver-side reader share one key set, pinned end to end."""
        config_path: Path = tmp_path / "sglang.yaml"
        config_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: decode\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 2\n"
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, sglang_api_key="from-args")
        (spec,) = specs_inference_engine(args)

        info = CellInfo(
            cell_id="inference-engine-0-0-1",
            pool_id=spec.name,
            alive=True,
            worker_names=["inference-engine-0-0-1-0"],
            workers_hash="pseudo-hash-0",
            meta=spec.meta(WorkerMetaContext(cell_index=1)),
        )

        assert _compute_server_cell_meta_from_info(info) == ServerCellMetadata(
            model_id="default",
            worker_type="decode",
            cell_id="inference-engine-0-0-1",
            num_gpus_per_engine=2,
            gpu_offset=2,
            sglang_api_key="from-args",
            worker_name="inference-engine-0-0-1-0",
            needs_offload=False,
            update_weights=True,
            workers_hash="pseudo-hash-0",
        )


class TestUpdateWeightsLockWindow:
    @pytest.mark.asyncio
    async def test_the_lock_is_held_from_start_until_end_update_weights(self):
        """start_update_weights opens a lock window that only end_update_weights closes."""
        controller = _make_controller({})

        info = await controller.start_update_weights()
        assert controller.context_lock.locked

        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
        assert not controller.context_lock.locked

    @pytest.mark.asyncio
    async def test_reconcile_waits_while_the_update_weights_window_is_open(self):
        """A concurrent reconcile must not mutate the engine set mid weight update."""
        controller = _make_controller({})
        info = await controller.start_update_weights()

        reconcile_task = asyncio.create_task(controller._reconcile("miles-router-0-0", None))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not reconcile_task.done()

        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
        await reconcile_task

    @pytest.mark.asyncio
    async def test_a_plain_locked_call_does_not_leave_the_lock_held(self):
        """Ordinary controller methods release the lock when they return."""
        controller = _make_controller({})
        await controller.prepare_eval()
        assert not controller.context_lock.locked


class TestServersShareTheControllerLock:
    @pytest.mark.asyncio
    async def test_reconcile_can_drive_the_server_it_owns(self):
        """The controller lock is the very lock its servers require, so reconcile works end to end."""
        controller = _make_controller({})
        srv = RolloutServer(
            server_cells={},
            args=SimpleNamespace(),
            context_lock=controller.context_lock,
            engine_provider=_FakeWorkerProvider([]),
        )
        controller.servers = {"default": srv}
        info = _make_cell_info()

        await controller._reconcile(info.cell_id, None)
        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_server_holding_a_foreign_lock_is_rejected(self):
        """A server wired up with its own lock instead of the controller's is a wiring bug."""
        controller = _make_controller({})
        srv = RolloutServer(
            server_cells={},
            args=SimpleNamespace(),
            context_lock=ContextLock("InferenceController"),
            engine_provider=_FakeWorkerProvider([]),
        )
        controller.servers = {"default": srv}

        with pytest.raises(AssertionError, match="must be called with"):
            await controller.offload()


class TestUpdatableModelSelection:
    @staticmethod
    def _controller(*servers: _RecordingServer) -> InferenceController:
        return _make_controller({srv.model_name: srv for srv in servers})

    @pytest.mark.asyncio
    async def test_only_the_updatable_models_engines_receive_weights(self):
        """A frozen reference model handed the trainer's weights stops being the baseline the
        KL term is measured against."""
        actor = _RecordingServer(model_name="actor", update_weights=True)
        actor.api_clients = ["actor-client"]
        ref = _RecordingServer(model_name="ref", update_weights=False)
        ref.api_clients = ["ref-client"]

        updatable = await self._controller(actor, ref).start_update_weights()

        assert updatable.rollout_engines == ["actor-client"]

    @pytest.mark.asyncio
    async def test_an_inference_only_deployment_updates_nothing(self):
        """No model is being trained, so there is no engine to push weights into; returning a
        frozen model's engines here would overwrite it."""
        updatable = await self._controller(_RecordingServer(model_name="ref")).start_update_weights()

        assert updatable.rollout_engines == []
        assert updatable.snapshot_cell_id_to_hashes == {}

    @pytest.mark.asyncio
    async def test_two_updatable_models_are_refused_by_name(self):
        """Picking one arbitrarily would silently train one model and leave the other stale."""
        controller = self._controller(
            _RecordingServer(model_name="a", update_weights=True),
            _RecordingServer(model_name="b", update_weights=True),
        )

        with pytest.raises(ValueError, match="Multiple servers have update_weights=True"):
            await controller.start_update_weights()

    @pytest.mark.asyncio
    async def test_the_ambiguity_points_at_the_way_out_of_it(self):
        """Naming a model id is the supported way to update one of several trainable models."""
        controller = self._controller(
            _RecordingServer(model_name="a", update_weights=True),
            _RecordingServer(model_name="b", update_weights=True),
        )

        with pytest.raises(ValueError, match="Pass model_id to update exactly one of them"):
            await controller.start_update_weights()

    @pytest.mark.asyncio
    async def test_the_weight_checker_skips_the_frozen_models(self):
        """reset_tensors on a model nobody will rewrite scrambles it for the rest of the run."""
        actor = _RecordingServer(model_name="actor", update_weights=True)
        ref = _RecordingServer(model_name="ref", update_weights=False)

        assert await self._controller(actor, ref).check_weights(action="snapshot") == ["actor"]
        assert ref.calls == []

    @pytest.mark.asyncio
    async def test_the_weight_checker_is_a_noop_without_an_updatable_model(self):
        """Nothing was updated, so there is nothing to compare against."""
        ref = _RecordingServer(model_name="ref")

        assert await self._controller(ref).check_weights(action="compare") == []
        assert ref.calls == []

    @pytest.mark.asyncio
    async def test_check_weights_forwards_all_selection_arguments(self):
        """Losing the selector or the skip list here would compare tensors the caller asked to leave alone."""
        actor = _RecordingServer(model_name="actor", update_weights=True)

        await self._controller(actor).check_weights(
            action="compare", allow_quant_error=True, selector="first", skip_list=["lm_head"]
        )

        assert actor.check_weights_kwargs == [
            dict(action="compare", allow_quant_error=True, selector="first", skip_list=["lm_head"])
        ]


class TestGetServersOfModelId:
    @staticmethod
    def _controller(*servers: _RecordingServer) -> InferenceController:
        return _make_controller({srv.model_name: srv for srv in servers})

    @pytest.mark.asyncio
    async def test_a_named_policy_scopes_the_update_to_its_own_server(self):
        """Several policies train at once, so the ambiguity a single-policy run never hits is the normal case."""
        alpha = _RecordingServer(model_name="alpha", update_weights=True)
        alpha.api_clients = ["alpha-client"]
        beta = _RecordingServer(model_name="beta", update_weights=True)
        beta.api_clients = ["beta-client"]

        updatable = await self._controller(alpha, beta).start_update_weights(model_id="beta")

        assert updatable.rollout_engines == ["beta-client"]

    @pytest.mark.asyncio
    async def test_an_unknown_model_id_is_refused(self):
        """Guessing a server here would push one policy's weights into another policy's engines."""
        controller = self._controller(
            _RecordingServer(model_name="alpha", update_weights=True),
            _RecordingServer(model_name="beta", update_weights=True),
        )

        with pytest.raises(AssertionError, match=r"No server for model_id 'nope'.*\['alpha', 'beta'\]"):
            await controller.start_update_weights(model_id="nope")

    @pytest.mark.asyncio
    async def test_a_frozen_server_named_explicitly_is_refused(self):
        """A reference model is frozen on purpose; overwriting it destroys the baseline the KL term uses."""
        controller = self._controller(
            _RecordingServer(model_name="alpha", update_weights=True),
            _RecordingServer(model_name="ref", update_weights=False),
        )

        with pytest.raises(AssertionError, match="Server for model_id 'ref' is frozen"):
            await controller.start_update_weights(model_id="ref")

    @pytest.mark.asyncio
    async def test_the_weight_checker_compares_only_the_named_policys_engines(self):
        """A checksum taken against the other policy's engines would report a mismatch on every step."""
        alpha = _RecordingServer(model_name="alpha", update_weights=True)
        beta = _RecordingServer(model_name="beta", update_weights=True)

        assert await self._controller(alpha, beta).check_weights(action="checksum", model_id="alpha") == ["alpha"]
        assert beta.calls == []


class TestEnsureCellsReady:
    @staticmethod
    def _controller() -> InferenceController:
        """One policy is serving, the other still has a cell coming up."""
        ready = _FakeUpdatableCell("hash-a")
        pending = _FakeUpdatableCell("hash-b")
        pending.is_pending_weights_or_serving = False
        return _make_controller(
            {
                "alpha": _RecordingServer({"alpha-0": ready}, model_name="alpha", update_weights=True),
                "beta": _RecordingServer({"beta-0": pending}, model_name="beta", update_weights=False),
            }
        )

    @pytest.mark.asyncio
    async def test_only_the_cells_of_the_named_policy_are_waited_for(self, monkeypatch):
        """One policy must not be blocked from updating by another policy's cells still coming up."""
        monkeypatch.setattr(inference_controller_module, "CELLS_READY_TIMEOUT_SECONDS", 0)

        updatable = await self._controller().start_update_weights(model_id="alpha")

        assert updatable.snapshot_cell_id_to_hashes == {"alpha-0": "hash-a"}

    @pytest.mark.asyncio
    async def test_an_unscoped_update_still_waits_for_every_cell_of_the_run(self, monkeypatch):
        """The scope is what makes the wait short, so the unscoped path must keep covering all engines."""
        monkeypatch.setattr(inference_controller_module, "CELLS_READY_TIMEOUT_SECONDS", 0)

        with pytest.raises(TimeoutError, match="waiting for 1/2 cells"):
            await self._controller().start_update_weights()


class TestMemoryLifecycleFanOut:
    @pytest.mark.asyncio
    async def test_memory_lifecycle_entrypoints_fan_out_with_exact_tags(self):
        """Every server must be told exactly which memory pools to release and to reclaim."""
        first, second = _RecordingServer(model_name="a"), _RecordingServer(model_name="b")
        controller = _make_controller({"a": first, "b": second})

        await controller.offload(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        await controller.onload(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
        await controller.onload_weights()
        await controller.onload_kv()

        for srv in (first, second):
            assert srv.offload_tags == [[GPU_MEMORY_TYPE_KV_CACHE]]
            assert srv.onload_tags == [
                [GPU_MEMORY_TYPE_CUDA_GRAPH],
                [GPU_MEMORY_TYPE_WEIGHTS],
                [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH],
            ]


class TestUpdatableEnginesPayload:
    @pytest.mark.asyncio
    async def test_start_update_weights_returns_clients_gpu_layout_and_generation_snapshot(self):
        """The trainer indexes these four lists in parallel, so swapping or dropping one misplaces every shard."""
        srv = _RecordingServer(
            {"engine-0": _FakeUpdatableCell("hash-a"), "engine-1": _FakeUpdatableCell("hash-b")},
            model_name="actor",
            update_weights=True,
        )
        srv.api_clients = ["client-0", "client-1"]
        srv.engine_gpu_counts = [2, 4]
        srv.engine_gpu_offsets = [0, 2]
        controller = _make_controller({"actor": srv, "ref": _RecordingServer(model_name="ref")})

        updatable = await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes=updatable.snapshot_cell_id_to_hashes)

        assert updatable == UpdatableEngines(
            rollout_engines=["client-0", "client-1"],
            engine_gpu_counts=[2, 4],
            engine_gpu_offsets=[0, 2],
            snapshot_cell_id_to_hashes={"engine-0": "hash-a", "engine-1": "hash-b"},
        )

    @pytest.mark.asyncio
    async def test_end_update_weights_skips_a_cell_from_a_different_worker_generation(self):
        """A cell relaunched during the update runs new processes that never received these weights."""
        relaunched, untouched = _FakeUpdatableCell("hash-new"), _FakeUpdatableCell("hash-b")
        srv = _RecordingServer(
            {"engine-0": relaunched, "engine-1": untouched}, model_name="actor", update_weights=True
        )
        controller = _make_controller({"actor": srv})

        await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes={"engine-0": "hash-old", "engine-1": "hash-b"})

        assert (relaunched.marked_ready, untouched.marked_ready) == (0, 1)


class TestInitLifecycle:
    @staticmethod
    def _controller(args: Namespace, *, engine_provider: _FakeWorkerProvider | None = None) -> InferenceController:
        return InferenceController(
            args,
            engine_provider=engine_provider if engine_provider is not None else _FakeWorkerProvider([]),
            router_providers=[_FakeWorkerProvider([])],
        )

    @pytest.mark.asyncio
    async def test_debug_train_only_init_has_no_rollout_side_effects(self, monkeypatch: pytest.MonkeyPatch):
        """A train-only debug run owns no engines, so init must not reach any rollout machinery."""

        async def _no_servers(args: Namespace, **kwargs: Any) -> dict:
            raise AssertionError("debug_train_only must not create rollout servers")

        async def _no_router_addrs(args: Namespace, **kwargs: Any) -> dict:
            raise AssertionError("debug_train_only must not resolve any router")

        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _no_servers)
        monkeypatch.setattr(inference_controller_module, "resolve_router_addrs", _no_router_addrs)
        monkeypatch.setattr(
            dashboard_hooks, "register_router", lambda args: pytest.fail("debug_train_only has no router")
        )
        provider = _RefusingWorkerProvider()
        controller = InferenceController(
            make_args(debug_train_only=True),
            engine_provider=provider,
            router_providers=[_RefusingWorkerProvider()],
        )

        await controller.init()

        assert controller.servers == {}
        assert controller._eval_fleet is None
        assert controller._watcher_disposers == []
        assert controller._ticker is None
        assert provider.initialized is False

    @pytest.mark.asyncio
    async def test_init_passes_its_exact_context_lock_to_the_server_factory(self, monkeypatch: pytest.MonkeyPatch):
        """A server built on a second lock would let engine work run inside the controller's own window."""
        received: dict[str, Any] = {}

        async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
            received.update(kwargs)
            return {"default": _RecordingServer()}

        _patch_init(monkeypatch, servers={"default": _RecordingServer()})
        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
        controller = self._controller(make_args())

        await controller.init()
        await controller.dispose()

        assert received["context_lock"] is controller.context_lock

    @pytest.mark.asyncio
    async def test_init_creates_the_eval_fleet_from_the_eval_server(self, monkeypatch: pytest.MonkeyPatch):
        """The eval fleet drives the dedicated eval engines, so it must be handed that server and no other."""
        monkeypatch.setattr(inference_controller_module, "InferenceControllerEvalFleet", _RecordingEvalFleet)
        default, eval_srv = _RecordingServer(model_name="default"), _RecordingServer(model_name="eval")
        _patch_init(monkeypatch, servers={"default": default, "eval": eval_srv})
        controller = self._controller(make_args(eval_num_gpus=2))

        await controller.init()
        await controller.dispose()

        assert isinstance(controller._eval_fleet, _RecordingEvalFleet)
        assert controller._eval_fleet.srv is eval_srv
        assert controller._eval_fleet.args is controller.args

    @pytest.mark.asyncio
    async def test_init_without_eval_gpus_creates_no_eval_fleet(self, monkeypatch: pytest.MonkeyPatch):
        """A run without dedicated eval engines has no eval server to build a fleet from."""
        monkeypatch.setattr(
            inference_controller_module,
            "InferenceControllerEvalFleet",
            lambda *args, **kwargs: pytest.fail("no eval fleet without eval gpus"),
        )
        _patch_init(monkeypatch, servers={"default": _RecordingServer()})
        controller = self._controller(make_args(eval_num_gpus=0))

        await controller.init()
        await controller.dispose()

        assert controller._eval_fleet is None

    @pytest.mark.asyncio
    async def test_init_registers_routing_and_waits_for_every_startup_gate(self, monkeypatch: pytest.MonkeyPatch):
        """Returning before every server has its cells would start a rollout against engines that are not up."""
        registered: list[Namespace] = []
        monkeypatch.setattr(dashboard_hooks, "register_router", registered.append)
        gate = asyncio.Event()
        ready, blocked = _RecordingServer(), _RecordingServer(cells_gate=gate)
        _patch_init(monkeypatch, servers={"default": ready, "frozen": blocked})
        args = make_args()
        controller = self._controller(args)

        task = asyncio.create_task(controller.init())
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done()
        assert ready.waited_init_expected_num_cells == 1
        gate.set()
        await asyncio.wait_for(task, timeout=5)
        await controller.dispose()

        assert registered == [args]
        assert blocked.waited_init_expected_num_cells == 1

    @pytest.mark.asyncio
    async def test_init_does_not_report_itself_initialized_before_its_fleet_is_in(self, monkeypatch):
        """A take-over reads this answer, and a controller still short of its cells is not one to take over."""
        gate = asyncio.Event()
        _patch_init(monkeypatch, servers={"frozen": _RecordingServer(cells_gate=gate)})
        controller = self._controller(make_args())

        task = asyncio.create_task(controller.init())
        for _ in range(20):
            await asyncio.sleep(0)
        assert await controller.is_initialized() is False

        gate.set()
        await asyncio.wait_for(task, timeout=5)
        assert await controller.is_initialized() is True

        await controller.dispose()

    @pytest.mark.asyncio
    async def test_init_and_dispose_own_the_cell_watch_and_ticker_lifetimes(self, monkeypatch: pytest.MonkeyPatch):
        """A watch or tick loop outliving the controller keeps dialing engines that nobody owns any more."""
        monkeypatch.setattr(inference_controller_module, "TICK_INTERVAL_SECONDS", 0.01)
        cell = _TickingCell()
        provider = _FakeWorkerProvider([])
        _patch_init(monkeypatch, servers={"default": _RecordingServer({"engine-0": cell})})
        controller = self._controller(make_args(), engine_provider=provider)

        await controller.init()
        await asyncio.sleep(0.05)
        assert cell.tick_count > 0
        assert controller._ticker._interval_seconds == inference_controller_module.TICK_INTERVAL_SECONDS
        assert len(controller._watcher_disposers) == 1

        await controller.dispose()
        ticks_at_dispose = cell.tick_count
        await asyncio.sleep(0.01)

        assert provider.stop_watch_calls == 1
        assert controller._watcher_disposers == []
        assert controller._ticker is None
        assert cell.tick_count == ticks_at_dispose


async def _raise_async(cell: ServerCell) -> None:
    raise RuntimeError("injected init failure")


_ROLLOUT_CELL_ID = "inference-engine-0-0-0"
_SPARE_ROLLOUT_CELL_ID = "inference-engine-0-1-0"


class _StoppingWorkerProvider(_FakeWorkerProvider):
    def __init__(self, *, completion: asyncio.Future | None = None) -> None:
        super().__init__([])
        self.stopped_cells: list[list[str]] = []
        self.stop_requested = asyncio.Event()
        self._completion = completion

    async def stop_cells(self, *, cell_ids: list[str]) -> None:
        self.stopped_cells.append(list(cell_ids))
        self.stop_requested.set()
        if self._completion is not None:
            await self._completion


def _make_cell_operations_controller(
    provider: _StoppingWorkerProvider, *, probing_paused: bool = False
) -> InferenceController:
    server = _RecordingServer(server_cells={_ROLLOUT_CELL_ID: object(), _SPARE_ROLLOUT_CELL_ID: object()})
    if probing_paused:
        server.health_checker_activeness.bump_active(False)
    return _make_controller({"default": server}, engine_provider=provider)


async def _hold_the_context_lock(controller: InferenceController) -> tuple[asyncio.Task, asyncio.Event]:
    entered, may_finish = asyncio.Event(), asyncio.Event()

    async def _hold() -> None:
        async with controller.context_lock:
            entered.set()
            await may_finish.wait()

    holder = asyncio.create_task(_hold())
    await entered.wait()
    return holder, may_finish


class TestCellOperations:
    @pytest.mark.asyncio
    async def test_stop_cell_between_weight_updates_is_forwarded_to_the_engine_provider(self):
        """The provider owns the processes, so the controller only serializes the suspension."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)

        await controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID)

        assert provider.stopped_cells == [[_ROLLOUT_CELL_ID]]

    @pytest.mark.asyncio
    async def test_a_suspended_cell_stops_being_addressable_before_the_lock_is_released(self):
        """A suspension is a kill too, so the next broadcast must not snapshot the cell it tore down."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)

        await controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID)

        assert controller.servers["default"].faulted_cells == [_ROLLOUT_CELL_ID]

    @pytest.mark.asyncio
    async def test_stop_cell_between_weight_updates_waits_until_the_weight_update_window_closes(self):
        """Suspending a cell mid-broadcast leaves the trainer waiting on an engine that is being torn down."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)
        holder, may_finish = await _hold_the_context_lock(controller)

        stopping = asyncio.create_task(controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID))
        await asyncio.sleep(0)

        assert provider.stopped_cells == []

        may_finish.set()
        await holder
        await stopping

        assert provider.stopped_cells == [[_ROLLOUT_CELL_ID]]

    @pytest.mark.asyncio
    async def test_inject_fault_between_weight_updates_waits_until_the_weight_update_window_closes(self):
        """Injection racing a broadcast is the same hazard as suspension, so it takes the same turn."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)
        holder, may_finish = await _hold_the_context_lock(controller)

        injecting = asyncio.create_task(
            controller.inject_fault_between_weight_updates(_ROLLOUT_CELL_ID, mode=FailureMode.SIGKILL, sub_index=0)
        )
        await asyncio.sleep(0)

        assert provider._worker_manager_handle.inject_fault.calls == []

        may_finish.set()
        await holder
        await injecting

        assert provider._worker_manager_handle.inject_fault.calls == [
            ((_ROLLOUT_CELL_ID,), {"mode": "sigkill", "worker_in_cell_index": 0, "wait_until_applied": True})
        ]

    @pytest.mark.asyncio
    async def test_stop_cell_between_weight_updates_is_allowed_while_probing_is_paused(self):
        """An offloaded cell is the one a heal loop most needs to suspend, so only injection is refused."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider, probing_paused=True)

        await controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID)

        assert provider.stopped_cells == [[_ROLLOUT_CELL_ID]]

    @pytest.mark.asyncio
    async def test_inject_fault_between_weight_updates_refuses_a_pause_that_began_while_it_waited(self):
        """Reading the pause before taking the lock would kill a cell the offload has since put to sleep."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider)
        server = controller.servers["default"]
        entered, may_finish = asyncio.Event(), asyncio.Event()

        async def _pause_probing_under_the_lock() -> None:
            async with controller.context_lock:
                entered.set()
                await may_finish.wait()
                server.health_checker_activeness.bump_active(False)

        holder = asyncio.create_task(_pause_probing_under_the_lock())
        await entered.wait()
        injecting = asyncio.create_task(
            controller.inject_fault_between_weight_updates(_ROLLOUT_CELL_ID, mode=FailureMode.SIGKILL, sub_index=0)
        )
        await asyncio.sleep(0)
        may_finish.set()
        await holder

        with pytest.raises(RuntimeError, match="refusing fault injection"):
            await injecting

        assert provider._worker_manager_handle.inject_fault.calls == []

    @pytest.mark.asyncio
    async def test_a_refused_injection_leaves_the_weight_update_lock_free(self):
        """A refusal that kept the lock would hang the next weight update instead of only skipping the injection."""
        provider = _StoppingWorkerProvider()
        controller = _make_cell_operations_controller(provider, probing_paused=True)

        with pytest.raises(RuntimeError, match="refusing fault injection"):
            await controller.inject_fault_between_weight_updates(
                _ROLLOUT_CELL_ID, mode=FailureMode.SIGKILL, sub_index=0
            )

        assert not controller.context_lock.locked

        await controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID)

        assert provider.stopped_cells == [[_ROLLOUT_CELL_ID]]

    @pytest.mark.asyncio
    async def test_a_weight_update_cannot_start_while_a_suspension_is_still_running(self):
        """Releasing the lock before the provider has torn the cell down reopens the very race this serializes."""
        completion: asyncio.Future = asyncio.get_running_loop().create_future()
        provider = _StoppingWorkerProvider(completion=completion)
        controller = _make_cell_operations_controller(provider)
        weight_update_started = asyncio.Event()

        async def _start_weight_update() -> None:
            async with controller.context_lock:
                weight_update_started.set()

        stopping = asyncio.create_task(controller.stop_cell_between_weight_updates(_ROLLOUT_CELL_ID))
        await provider.stop_requested.wait()
        weight_update = asyncio.create_task(_start_weight_update())
        await asyncio.sleep(0)

        assert not weight_update_started.is_set()

        completion.set_result(None)
        await stopping
        await weight_update

        assert weight_update_started.is_set()


def _raise_configure_logger(*args, **kwargs):
    raise ValueError("configure_logger blew up")


class TestInitRunsExactlyOnce:
    @staticmethod
    def _controller() -> InferenceController:
        return InferenceController(
            make_args(debug_train_only=True),
            engine_provider=_FakeWorkerProvider([]),
            router_providers=[_FakeWorkerProvider([])],
        )

    @pytest.mark.asyncio
    async def test_a_controller_that_never_ran_init_reports_itself_uninitialized(self):
        """A restarted script asks the controller it found running whether to initialize it or to take it over."""
        assert await self._controller().is_initialized() is False

    @pytest.mark.asyncio
    async def test_a_controller_that_ran_init_reports_itself_initialized(self):
        """The take-over path has to see the controller the previous script built as built."""
        controller = self._controller()

        await controller.init()

        assert await controller.is_initialized() is True

    @pytest.mark.asyncio
    async def test_a_second_init_is_refused(self):
        """Re-initializing a live controller would rebuild a fleet the previous script is still driving."""
        controller = self._controller()
        await controller.init()

        with pytest.raises(AssertionError, match="stale worker"):
            await controller.init()

    @pytest.mark.asyncio
    async def test_a_controller_whose_init_raised_reports_itself_uninitialized(self, monkeypatch):
        """A controller with no watcher and no ticker must not be taken over as a built one."""
        controller = self._controller()
        monkeypatch.setattr(inference_controller_module, "configure_logger", _raise_configure_logger)

        with pytest.raises(ValueError, match="configure_logger blew up"):
            await controller.init()

        assert await controller.is_initialized() is False

    @pytest.mark.asyncio
    async def test_a_controller_whose_init_raised_still_refuses_a_second_init(self, monkeypatch):
        """A half-built controller cannot be rebuilt over its own state, so it has to fail loudly instead."""
        controller = self._controller()
        monkeypatch.setattr(inference_controller_module, "configure_logger", _raise_configure_logger)
        with pytest.raises(ValueError):
            await controller.init()
        monkeypatch.undo()

        with pytest.raises(AssertionError, match="stale worker"):
            await controller.init()

    @pytest.mark.asyncio
    async def test_a_second_init_is_refused_after_a_full_init(self, monkeypatch: pytest.MonkeyPatch):
        """The train-only shortcut returns early, so the refusal has to hold for a controller that built a fleet."""
        _patch_init(monkeypatch, servers={"default": _RecordingServer()})
        controller = InferenceController(
            make_args(), engine_provider=_FakeWorkerProvider([]), router_providers=[_FakeWorkerProvider([])]
        )
        await controller.init()

        with pytest.raises(AssertionError):
            await controller.init()

        await controller.dispose()


class TestWaitingForTheWholeFleet:
    @pytest.mark.asyncio
    async def test_the_budget_the_caller_gives_reaches_every_server(self):
        """A take-over waits inside its own bounded gate, and a server ignoring that budget hangs the whole gate."""
        controller = _make_controller({"a": _RecordingServer(model_name="a"), "b": _RecordingServer(model_name="b")})

        await controller.wait_expected_num_cells(timeout=12.0)

        assert [srv.cells_timeouts for srv in controller.servers.values()] == [[12.0], [12.0]]

    @pytest.mark.asyncio
    async def test_a_caller_that_names_no_budget_gets_the_fleet_wide_one(self):
        """Init waits without naming a budget, and an unbounded wait there hangs the run with no diagnosis."""
        controller = _make_controller({"a": _RecordingServer(model_name="a")})

        await controller.wait_expected_num_cells()

        assert [srv.cells_timeouts for srv in controller.servers.values()] == [
            [inference_controller_module.CELLS_READY_TIMEOUT_SECONDS]
        ]


class TestEvalFleetSurface:
    def test_the_eval_fleet_is_an_rpc_method_rather_than_an_attribute(self):
        """A handle resolves rpc methods only, so reading the fleet off it reaches nothing."""
        handle = RpcWorkerHandle(InferenceController, server_url="http://10.0.0.1:1234")

        assert callable(handle.get_eval_fleet_info)
        assert callable(handle.pin_eval_fleet)
        with pytest.raises(AttributeError, match="no rpc method 'eval_fleet'"):
            handle.eval_fleet()

    def test_the_fleet_description_survives_the_wire(self):
        """The executor retargets its eval args to what it decodes, so every field must round-trip."""
        serializer = collect_rpc_method_specs(InferenceController)["get_eval_fleet_info"].serializer
        info = EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1)

        assert serializer.decode_result(serializer.encode_result(info)) == info
        assert serializer.decode_result(serializer.encode_result(None)) is None

    def test_a_pin_and_its_skip_reason_survive_the_wire(self):
        """A skipped point must arrive as a skip with its reason, not as a remote crash."""
        spec = collect_rpc_method_specs(InferenceController)["pin_eval_fleet"]
        query = dict(checkpoint_dir="/snap/step_5", weight_version="5")

        assert spec.serializer.decode_query(spec.serializer.encode_query(query)) == query
        for pin in (EvalFleetPin(skip_reason=None), EvalFleetPin(skip_reason="unhealthy")):
            assert spec.serializer.decode_result(spec.serializer.encode_result(pin)) == pin

    @pytest.mark.asyncio
    async def test_a_run_without_a_fleet_answers_nothing_to_wire_up(self):
        """--eval-num-gpus 0 deploys no fleet, and the executor must be told so rather than guess."""
        controller = _make_controller({})
        controller._eval_fleet = None

        assert await controller.get_eval_fleet_info() is None

    @pytest.mark.asyncio
    async def test_pinning_a_fleet_that_is_not_deployed_is_refused(self):
        """Nobody can reach this call without a fleet, so answering a skip would hide a wiring bug."""
        controller = _make_controller({})
        controller._eval_fleet = None

        with pytest.raises(AssertionError, match="no eval fleet"):
            await controller.pin_eval_fleet(checkpoint_dir="/snap/step_5", weight_version="5")

    @pytest.mark.asyncio
    async def test_the_fleet_answers_and_pins_through_the_controller(self):
        """The fleet lives beside its engines: the executor only ever addresses it through the controller."""
        controller = _make_controller({})
        info = EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1)
        controller._eval_fleet = _RecordingInferenceControllerEvalFleet(info)

        assert await controller.get_eval_fleet_info() == info
        assert await controller.pin_eval_fleet(checkpoint_dir="/snap/step_5", weight_version="5") == EvalFleetPin(
            skip_reason=None
        )
        assert controller._eval_fleet.pins == [dict(checkpoint_dir="/snap/step_5", weight_version="5")]


class TestCellsReadyIsScopedToTheTargetedModel:
    @pytest.mark.asyncio
    async def test_a_named_model_does_not_wait_for_another_models_cells(self):
        """Different model ids are independent, so a sick engine of B must not stall A's weight update."""
        a = _RecordingServer(model_name="a", update_weights=True)
        a.api_clients = ["a-client"]
        a.server_cells = {
            "a-0": SimpleNamespace(
                is_pending_weights_or_serving=True,
                is_uninitialized=False,
                meta=SimpleNamespace(workers_hash="hash-a"),
            )
        }
        b = _RecordingServer(model_name="b", update_weights=True)
        b.server_cells = {
            "b-0": SimpleNamespace(
                is_pending_weights_or_serving=False,
                is_uninitialized=False,
                meta=SimpleNamespace(workers_hash="hash-b"),
            )
        }
        controller = _make_controller({"a": a, "b": b})

        updatable = await controller.start_update_weights(model_id="a")

        assert updatable.rollout_engines == ["a-client"]

    @pytest.mark.asyncio
    async def test_an_unnamed_update_still_waits_for_every_model(self):
        """A single policy run has one server, so scoping must not change what it waits for."""
        srv = _RecordingServer(model_name="a", update_weights=True)
        controller = _make_controller({"a": srv})

        async with controller.context_lock:
            assert controller._get_servers_of_model_id(None) == [srv]
            assert controller._get_servers_of_model_id("a") == [srv]


def _registration_snapshot(*, model_id: str = "model-a", run_uuid: str = _RUN_UUID) -> RegistrationSnapshot:
    meta = dict(
        model_id=model_id,
        worker_type="regular",
        num_gpus_per_engine=1,
        gpu_offset=0,
        sglang_api_key=None,
        needs_offload=False,
        update_weights=True,
    )

    cell = RegisteredCellInfo(
        reporter_id="west",
        info=CellInfo(
            cell_id="west-inference-engine-0-0-00000",
            pool_id="west-inference-engine-0-0",
            alive=True,
            worker_names=["west-inference-engine-0-0-00000-00000"],
            workers_hash="hash-1",
            meta=meta,
        ),
        workers=[
            WorkerInfo(
                name="west-inference-engine-0-0-00000-00000",
                generation=0,
                self_addrs={"primary": HostAndPort(host="10.0.0.5", port=8000)},
                gpu_ids=[0],
            )
        ],
    )
    return RegistrationSnapshot(
        run_uuid=run_uuid,
        reporter_id="west",
        sequence_number=1,
        cells=[cell],
    )


class TestRegistrationSnapshotEndpoint:
    def test_a_snapshot_survives_the_wire(self):
        """The reporter is in another cluster, so the whole membership has to be wire typed both ways."""
        spec = collect_rpc_method_specs(InferenceController)["registration_ingest"]
        query = dict(snapshot=_registration_snapshot())

        assert spec.serializer.decode_query(spec.serializer.encode_query(query)) == query

    @pytest.mark.asyncio
    async def test_a_run_holding_a_registry_takes_the_snapshot_in(self):
        """This endpoint is the only way an engine of another deployment ever joins the run."""
        registry = RegistrationHub(run_uuid=_RUN_UUID)
        controller = _make_controller({"model-a": _RecordingServer(model_name="model-a")}, registration_hub=registry)

        await controller.registration_ingest(snapshot=_registration_snapshot())

        assert sorted(registry._cell_of_id) == ["west-inference-engine-0-0-00000"]

    @pytest.mark.asyncio
    async def test_a_controller_the_script_has_not_initialized_yet_says_it_is_not_ready(self):
        """Until init this controller knows no model of the run, and blaming the reporter for that misleads."""
        registry = RegistrationHub(run_uuid=_RUN_UUID)
        controller = _make_controller({}, registration_hub=registry)

        with pytest.raises(AssertionError, match="not ready"):
            await controller.registration_ingest(snapshot=_registration_snapshot())

        assert registry._cell_of_id == {}

    @pytest.mark.asyncio
    async def test_a_run_serving_engines_of_its_own_refuses_a_snapshot(self):
        """It would take the cells in and never wait for them, so the reporter has to hear that it is unwanted."""
        controller = _make_controller({"model-a": _RecordingServer(model_name="model-a")})

        with pytest.raises(AssertionError, match="takes no registration"):
            await controller.registration_ingest(snapshot=_registration_snapshot())

    @pytest.mark.asyncio
    async def test_a_cell_serving_a_model_this_run_does_not_serve_is_refused(self):
        """No router of this run would ever send it a request, so counting it would stall the wait for cells."""
        registry = RegistrationHub(run_uuid=_RUN_UUID)
        controller = _make_controller({"model-a": _RecordingServer(model_name="model-a")}, registration_hub=registry)

        with pytest.raises(AssertionError, match="serves model 'other'"):
            await controller.registration_ingest(snapshot=_registration_snapshot(model_id="other"))
