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
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts, WorkerMetaContext


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
        self.waited_expected_num_cells = 0
        self.dispose_count = 0
        self._cells_gate = cells_gate

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

    async def wait_expected_num_cells(self) -> None:
        if self._cells_gate is not None:
            await self._cells_gate.wait()
        self.waited_expected_num_cells += 1


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
    def __init__(self, args: Namespace, *, api_clients: list, router_host: str, router_port: int) -> None:
        self.args = args
        self.api_clients = api_clients
        self.router_host = router_host
        self.router_port = router_port

    async def dispose(self) -> None:
        return None


class _FakeWorkerProvider(BaseWorkerProvider):
    def __init__(self, cell_infos: list[CellInfo], *, pool_ids: list[str] | None = None) -> None:
        self._cell_infos = cell_infos
        self._pools = pool_ids or []
        self.watched_pool_ids: list[str] | None = None
        self.initialized = False

    async def init(self) -> None:
        self.initialized = True

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"the controller must not ask this fake for {worker_name}")

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [[] for _ in cell_ids]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        assert self.initialized, "the controller must init the provider before observing its cells"
        self.watched_pool_ids = list(self._pools)
        for info in self._cell_infos:
            if info.pool_id in self._pools:
                await reconcile(info.cell_id, info)

        async def _stop_watch() -> None:
            return None

        return _stop_watch


def _make_controller(servers: dict, *, engine_provider: _FakeWorkerProvider | None = None) -> InferenceController:
    controller = InferenceController.__new__(InferenceController)
    controller.args = SimpleNamespace(debug_train_only=False, use_fault_tolerance=False, ci_test=False, colocate=False)
    controller.servers = servers
    controller.context_lock = ContextLock("InferenceController")
    controller._health_checker_activeness = ActivenessTracker(active=True)
    controller._engine_provider = engine_provider if engine_provider is not None else _FakeWorkerProvider([])
    controller._router_providers = [_FakeWorkerProvider([])]
    return controller


class TestHealthCheckerActiveness:
    @pytest.mark.asyncio
    async def test_offload_pauses_probing_before_putting_engines_to_sleep(self):
        """A slept engine cannot answer /health_generate, so probing must stop first."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})

        await controller.offload()

        assert not controller._health_checker_activeness.get().active
        assert srv.calls == [("offload",)]

    @pytest.mark.asyncio
    async def test_starting_a_weight_update_pauses_probing(self):
        """Engines are unusable while their weights are being replaced."""
        controller = _make_controller({"default": _RecordingServer()})

        info = await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        assert not controller._health_checker_activeness.get().active

    @pytest.mark.asyncio
    async def test_preparing_a_rollout_resumes_probing(self):
        """Probing comes back exactly when the engines start serving traffic again."""
        controller = _make_controller({"default": _RecordingServer()})
        controller._health_checker_activeness.bump_active(False)

        await controller.prepare_rollout(rollout_id=0)

        assert controller._health_checker_activeness.get().active

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
        controller = _make_controller({"default": _RecordingServer()})
        controller._health_checker_activeness.bump_active(False)

        await controller.prepare_eval()

        assert controller._health_checker_activeness.get().active


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

        assert controller.get_cell_statuses() == {}


class TestGlobalHealthCheckerActiveness:
    @pytest.mark.asyncio
    async def test_init_hands_the_cells_the_controller_wide_activeness(self, monkeypatch: pytest.MonkeyPatch):
        """Without it every cell keeps probing through the weight-update window the controller
        just paused, and reports a mid-update engine unhealthy."""
        received: dict[str, Any] = {}

        async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
            received.update(kwargs)
            return {"default": _RecordingServer()}

        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
        monkeypatch.setattr(
            inference_controller_module,
            "RayWorkerProvider",
            SimpleNamespace(create=lambda *, pool_ids: _FakeWorkerProvider([]).created_with(pool_ids)),
        )
        controller = InferenceController(make_args())

        await controller.init()

        get_activeness = received["global_health_checker_activeness"]
        assert get_activeness().active is True
        controller._health_checker_activeness.bump_active(False)
        assert get_activeness().active is False


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

            async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
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
    async def test_init_survives_a_router_cell_offered_by_the_provider(self, monkeypatch: pytest.MonkeyPatch):
        """A router cell carries no engine meta, so a too-wide subscription kills startup in the initial sync."""
        args = make_args()
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
    @pytest.mark.asyncio
    async def test_debug_train_only_init_has_no_rollout_side_effects(self, monkeypatch: pytest.MonkeyPatch):
        """A train-only debug run owns no engines, so init must not reach any rollout machinery."""

        async def _no_servers(args: Namespace, **kwargs: Any) -> dict:
            raise AssertionError("debug_train_only must not create rollout servers")

        async def _no_session_server(args: Namespace) -> None:
            raise AssertionError("debug_train_only must not wait for the session server")

        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _no_servers)
        monkeypatch.setattr(inference_controller_module, "wait_session_server_ready", _no_session_server)
        monkeypatch.setattr(
            inference_controller_module,
            "RayWorkerProvider",
            SimpleNamespace(create=lambda **kwargs: pytest.fail("debug_train_only must not watch cells")),
        )
        monkeypatch.setattr(
            dashboard_hooks, "register_router", lambda args: pytest.fail("debug_train_only has no router")
        )
        controller = InferenceController(make_args(debug_train_only=True))

        await controller.init()

        assert controller.servers == {}
        assert controller.eval_fleet is None
        assert controller._watcher_disposers == []
        assert controller._ticker is None

    @pytest.mark.asyncio
    async def test_init_passes_its_exact_context_lock_to_the_server_factory(self, monkeypatch: pytest.MonkeyPatch):
        """A server built on a second lock would let engine work run inside the controller's own window."""
        received: dict[str, Any] = {}

        async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
            received.update(kwargs)
            return {"default": _RecordingServer()}

        monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
        monkeypatch.setattr(
            inference_controller_module,
            "RayWorkerProvider",
            SimpleNamespace(create=lambda *, pool_ids: _FakeWorkerProvider([]).created_with(pool_ids)),
        )
        controller = InferenceController(make_args())

        await controller.init()
        await controller.dispose()

        assert received["context_lock"] is controller.context_lock

    @pytest.mark.asyncio
    async def test_init_creates_the_eval_fleet_from_the_eval_server(self, monkeypatch: pytest.MonkeyPatch):
        """The eval fleet drives the dedicated eval engines, so it must be handed that server's clients and no other."""
        monkeypatch.setattr(inference_controller_module, "EvalFleet", _RecordingEvalFleet)
        default, eval_srv = _RecordingServer(model_name="default"), _RecordingServer(model_name="eval")
        eval_srv.api_clients = ["eval-client"]
        eval_srv.router_ip, eval_srv.router_port = "10.0.0.2", 31000
        _patch_init(monkeypatch, provider=_FakeWorkerProvider([]), servers={"default": default, "eval": eval_srv})
        controller = InferenceController(make_args(eval_num_gpus=2))

        await controller.init()
        await controller.dispose()

        assert isinstance(controller.eval_fleet, _RecordingEvalFleet)
        assert controller.eval_fleet.api_clients == ["eval-client"]
        assert (controller.eval_fleet.router_host, controller.eval_fleet.router_port) == ("10.0.0.2", 31000)
        assert controller.eval_fleet.args is controller.args

    @pytest.mark.asyncio
    async def test_init_without_eval_gpus_creates_no_eval_fleet(self, monkeypatch: pytest.MonkeyPatch):
        """A run without dedicated eval engines has no eval server to build a fleet from."""
        monkeypatch.setattr(
            inference_controller_module,
            "EvalFleet",
            lambda *args, **kwargs: pytest.fail("no eval fleet without eval gpus"),
        )
        _patch_init(monkeypatch, provider=_FakeWorkerProvider([]), servers={"default": _RecordingServer()})
        controller = InferenceController(make_args(eval_num_gpus=0))

        await controller.init()
        await controller.dispose()

        assert controller.eval_fleet is None

    @pytest.mark.asyncio
    async def test_init_registers_routing_and_waits_for_every_startup_gate(self, monkeypatch: pytest.MonkeyPatch):
        """Returning before every server has its cells would start a rollout against engines that are not up."""
        registered: list[Namespace] = []
        waited_session: list[Namespace] = []

        async def _wait_session_server_ready(args: Namespace) -> None:
            waited_session.append(args)

        monkeypatch.setattr(dashboard_hooks, "register_router", registered.append)
        monkeypatch.setattr(inference_controller_module, "wait_session_server_ready", _wait_session_server_ready)
        gate = asyncio.Event()
        ready, blocked = _RecordingServer(), _RecordingServer(cells_gate=gate)
        _patch_init(monkeypatch, provider=_FakeWorkerProvider([]), servers={"default": ready, "frozen": blocked})
        args = make_args()
        controller = InferenceController(args)

        task = asyncio.create_task(controller.init())
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done()
        assert ready.waited_expected_num_cells == 1
        gate.set()
        await asyncio.wait_for(task, timeout=5)
        await controller.dispose()

        assert registered == [args]
        assert waited_session == [args]
        assert blocked.waited_expected_num_cells == 1

    @pytest.mark.asyncio
    async def test_init_and_dispose_own_the_cell_watch_and_ticker_lifetimes(self, monkeypatch: pytest.MonkeyPatch):
        """A watch or tick loop outliving the controller keeps dialing engines that nobody owns any more."""
        monkeypatch.setattr(inference_controller_module, "TICK_INTERVAL_SECONDS", 0.01)
        cell = _TickingCell()
        provider = _FakeWorkerProvider([])
        _patch_init(monkeypatch, provider=provider, servers={"default": _RecordingServer({"engine-0": cell})})
        controller = InferenceController(make_args())

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


class _LateClientServer(_RecordingServer):
    async def wait_expected_num_cells(self) -> None:
        await super().wait_expected_num_cells()
        self.api_clients = ["eval-client-0"]


class _LockRecordingServer(_RecordingServer):
    def __init__(self, *, context_lock: ContextLock, **kwargs: Any) -> None:
        self._context_lock = context_lock
        self.lock_held_on_read: list[bool] = []
        super().__init__(**kwargs)
        self._api_clients = ["eval-client-0"]

    @property
    def api_clients(self) -> list:
        self.lock_held_on_read.append(self._context_lock.held_in_current_context)
        return self._api_clients

    @api_clients.setter
    def api_clients(self, value: list) -> None:
        self._api_clients = value


class TestEvalFleetConstruction:
    async def test_the_eval_fleet_is_built_only_once_the_servers_report_their_cells(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Built before the cells are up, the fleet snapshots an empty client list and every eval point pins nothing."""
        monkeypatch.setattr(inference_controller_module, "EvalFleet", _RecordingEvalFleet)
        eval_srv = _LateClientServer(model_name="eval")
        _patch_init(
            monkeypatch,
            provider=_FakeWorkerProvider([]),
            servers={"default": _RecordingServer(model_name="default"), "eval": eval_srv},
        )
        controller = InferenceController(make_args(eval_num_gpus=2))

        await controller.init()
        await controller.dispose()

        assert eval_srv.waited_expected_num_cells == 1
        assert controller.eval_fleet.api_clients == ["eval-client-0"]

    async def test_the_eval_server_clients_are_read_under_the_controller_context_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """RolloutServer.api_clients is lock-guarded, so an unlocked read trips the lock discipline in production."""
        monkeypatch.setattr(inference_controller_module, "EvalFleet", _RecordingEvalFleet)
        controller = InferenceController(make_args(eval_num_gpus=2))
        eval_srv = _LockRecordingServer(context_lock=controller.context_lock, model_name="eval")
        _patch_init(monkeypatch, provider=_FakeWorkerProvider([]), servers={"eval": eval_srv})

        await controller.init()
        await controller.dispose()

        assert eval_srv.lock_held_on_read == [True]

    async def test_the_eval_fleet_gets_a_copy_of_the_server_client_list(self, monkeypatch: pytest.MonkeyPatch):
        """Sharing the list would let a later cell change rewrite the snapshot the fleet already pinned against."""
        monkeypatch.setattr(inference_controller_module, "EvalFleet", _RecordingEvalFleet)
        eval_srv = _RecordingServer(model_name="eval")
        eval_srv.api_clients = ["eval-client-0"]
        _patch_init(monkeypatch, provider=_FakeWorkerProvider([]), servers={"eval": eval_srv})
        controller = InferenceController(make_args(eval_num_gpus=2))

        await controller.init()
        await controller.dispose()
        eval_srv.api_clients.append("late-client")

        assert controller.eval_fleet.api_clients == ["eval-client-0"]


async def _raise_async(cell: ServerCell) -> None:
    raise RuntimeError("injected init failure")


class _RecordingWorkerManager:
    def __init__(self) -> None:
        self.stopped_cells: list[list[str]] = []
        self.injected: list[tuple[str, dict[str, Any]]] = []

    @property
    def stop_cells(self) -> Any:
        return _RecordingRemoteCall(lambda cell_ids: self.stopped_cells.append(list(cell_ids)))

    @property
    def inject_fault(self) -> Any:
        return _RecordingRemoteCall(lambda cell_id, **kwargs: self.injected.append((cell_id, kwargs)))


class _RecordingRemoteCall:
    def __init__(self, record: Any) -> None:
        self._record = record

    def remote(self, *args: Any, **kwargs: Any) -> asyncio.Future:
        self._record(*args, **kwargs)
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        future.set_result(None)
        return future


def _patch_worker_manager(monkeypatch: pytest.MonkeyPatch) -> _RecordingWorkerManager:
    manager = _RecordingWorkerManager()
    monkeypatch.setattr(inference_controller_module, "RayWorkerManager", SimpleNamespace(get_handle=lambda: manager))
    return manager


class _BlockingWorkerManager:
    def __init__(self, *, completion: asyncio.Future) -> None:
        self.requested = asyncio.Event()
        self.stopped_cells: list[list[str]] = []
        self.completion = completion

    @property
    def stop_cells(self) -> Any:
        return _BlockingRemoteCall(manager=self)


class _BlockingRemoteCall:
    def __init__(self, *, manager: _BlockingWorkerManager) -> None:
        self._manager = manager

    def remote(self, cell_ids: list[str]) -> asyncio.Future:
        self._manager.stopped_cells.append(list(cell_ids))
        self._manager.requested.set()
        return self._manager.completion


def _patch_blocking_worker_manager(
    monkeypatch: pytest.MonkeyPatch, *, completion: asyncio.Future
) -> _BlockingWorkerManager:
    manager = _BlockingWorkerManager(completion=completion)
    monkeypatch.setattr(inference_controller_module, "RayWorkerManager", SimpleNamespace(get_handle=lambda: manager))
    return manager


class TestCellOperations:
    @pytest.mark.asyncio
    async def test_stop_cell_between_weight_updates_is_forwarded_to_the_worker_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The manager owns the processes, so the controller only serializes the suspension."""
        manager = _patch_worker_manager(monkeypatch)
        controller = _make_controller({})

        await controller.stop_cell_between_weight_updates("engine-0")

        assert manager.stopped_cells == [["engine-0"]]

    @pytest.mark.asyncio
    async def test_inject_fault_between_weight_updates_is_forwarded_to_the_worker_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Injection targets one worker of one cell, and only the manager can reach it."""
        manager = _patch_worker_manager(monkeypatch)
        controller = _make_controller({})

        await controller.inject_fault_between_weight_updates("engine-0", mode=FailureMode.SIGKILL, sub_index=1)

        assert manager.injected == [("engine-0", {"mode": "sigkill", "worker_in_cell_index": 1})]

    @pytest.mark.asyncio
    async def test_inject_fault_between_weight_updates_is_refused_while_probing_is_paused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """An offloaded or updating cell would report a crash that the trainer cannot distinguish from its own pause."""
        manager = _patch_worker_manager(monkeypatch)
        controller = _make_controller({})
        controller._health_checker_activeness.bump_active(False)

        with pytest.raises(RuntimeError, match="refusing fault injection"):
            await controller.inject_fault_between_weight_updates("engine-0", mode=FailureMode.SIGKILL, sub_index=0)

        assert manager.injected == []

    @pytest.mark.asyncio
    async def test_stop_cell_between_weight_updates_waits_until_the_weight_update_window_closes(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Suspending a cell mid-broadcast leaves the trainer waiting on an engine that is being torn down."""
        manager = _patch_worker_manager(monkeypatch)
        controller = _make_controller({})
        entered = asyncio.Event()
        may_finish = asyncio.Event()

        async def _hold_lock() -> None:
            async with controller.context_lock:
                entered.set()
                await may_finish.wait()

        holder = asyncio.create_task(_hold_lock())
        await entered.wait()
        stopping = asyncio.create_task(controller.stop_cell_between_weight_updates("engine-0"))
        await asyncio.sleep(0)

        assert manager.stopped_cells == []

        may_finish.set()
        await holder
        await stopping

        assert manager.stopped_cells == [["engine-0"]]

    @pytest.mark.asyncio
    async def test_inject_fault_between_weight_updates_waits_until_the_weight_update_window_closes(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Injection racing a broadcast is the same hazard as suspension, so it takes the same turn."""
        manager = _patch_worker_manager(monkeypatch)
        controller = _make_controller({})
        entered = asyncio.Event()
        may_finish = asyncio.Event()

        async def _hold_lock() -> None:
            async with controller.context_lock:
                entered.set()
                await may_finish.wait()

        holder = asyncio.create_task(_hold_lock())
        await entered.wait()
        injecting = asyncio.create_task(
            controller.inject_fault_between_weight_updates("engine-0", mode=FailureMode.SIGKILL, sub_index=0)
        )
        await asyncio.sleep(0)

        assert manager.injected == []

        may_finish.set()
        await holder
        await injecting

        assert manager.injected == [("engine-0", {"mode": "sigkill", "worker_in_cell_index": 0})]

    async def test_stop_cell_between_weight_updates_is_allowed_while_probing_is_paused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """An offloaded cell is the one a heal loop most needs to suspend, so only injection is refused."""
        manager = _patch_worker_manager(monkeypatch)
        controller = _make_controller({})
        controller._health_checker_activeness.bump_active(False)

        await controller.stop_cell_between_weight_updates("engine-0")

        assert manager.stopped_cells == [["engine-0"]]

    async def test_inject_fault_between_weight_updates_refuses_a_pause_that_began_while_it_waited(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Reading the pause before taking the lock would kill a cell the offload has since put to sleep."""
        manager = _patch_worker_manager(monkeypatch)
        controller = _make_controller({})
        entered = asyncio.Event()
        may_finish = asyncio.Event()

        async def _pause_probing_under_the_lock() -> None:
            async with controller.context_lock:
                entered.set()
                await may_finish.wait()
                controller._health_checker_activeness.bump_active(False)

        holder = asyncio.create_task(_pause_probing_under_the_lock())
        await entered.wait()
        injecting = asyncio.create_task(
            controller.inject_fault_between_weight_updates("engine-0", mode=FailureMode.SIGKILL, sub_index=0)
        )
        await asyncio.sleep(0)
        may_finish.set()
        await holder

        with pytest.raises(RuntimeError, match="refusing fault injection"):
            await injecting

        assert manager.injected == []

    async def test_a_refused_injection_leaves_the_weight_update_lock_free(self, monkeypatch: pytest.MonkeyPatch):
        """A refusal that kept the lock would hang the next weight update instead of only skipping the injection."""
        manager = _patch_worker_manager(monkeypatch)
        controller = _make_controller({})
        controller._health_checker_activeness.bump_active(False)

        with pytest.raises(RuntimeError, match="refusing fault injection"):
            await controller.inject_fault_between_weight_updates("engine-0", mode=FailureMode.SIGKILL, sub_index=0)

        assert not controller.context_lock.locked

        await controller.stop_cell_between_weight_updates("engine-0")

        assert manager.stopped_cells == [["engine-0"]]

    async def test_a_weight_update_cannot_start_while_a_suspension_is_still_running(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Releasing the lock before the manager has torn the cell down reopens the very race this serializes."""
        completion: asyncio.Future = asyncio.get_running_loop().create_future()
        manager = _patch_blocking_worker_manager(monkeypatch, completion=completion)
        controller = _make_controller({})
        weight_update_started = asyncio.Event()

        async def _start_weight_update() -> None:
            async with controller.context_lock:
                weight_update_started.set()

        stopping = asyncio.create_task(controller.stop_cell_between_weight_updates("engine-0"))
        await manager.requested.wait()
        weight_update = asyncio.create_task(_start_weight_update())
        await asyncio.sleep(0)

        assert not weight_update_started.is_set()

        completion.set_result(None)
        await stopping
        await weight_update

        assert weight_update_started.is_set()
