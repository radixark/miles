import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout.inference_controller import InferenceController, _compute_server_cell_meta_from_info
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCellMetadata
from miles.ray.specs.inference import compute_engine_pool_ids, compute_router_pool_id, specs_inference_engine
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.workers.worker_provider.base import CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, WorkerMetaContext


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
    def __init__(self, server_cells: dict | None = None):
        self.server_cells = server_cells or {}
        self.update_weights = False
        self.calls: list[tuple] = []

    async def offload(self, tags=None):
        self.calls.append(("offload",))

    async def add_cell(self, cell_meta: ServerCellMetadata):
        self.calls.append(("add", cell_meta.cell_id))
        self.server_cells[cell_meta.cell_id] = SimpleNamespace(meta=cell_meta)

    async def remove_cell(self, cell_id: str):
        self.calls.append(("remove", cell_id))
        del self.server_cells[cell_id]

    async def wait_expected_num_cells(self) -> None:
        return None

    async def dispose(self) -> None:
        return None


class _FakeWorkerProvider:
    def __init__(self, cell_infos: list[CellInfo], *, pool_ids: list[str] | None = None) -> None:
        self._cell_infos = cell_infos
        self._pools = pool_ids or []
        self.watched_pool_ids: list[str] | None = None

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
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
    controller._router_provider = _FakeWorkerProvider([])
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


async def _init_controller(args: Namespace, *, engine_provider: _FakeWorkerProvider) -> None:
    controller = InferenceController(args, engine_provider=engine_provider, router_provider=_FakeWorkerProvider([]))
    await controller.init()
    await controller.dispose()


class TestInitSubscription:
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
        engine_info = _make_cell_info(model_id="default")
        provider = _FakeWorkerProvider(
            [router_info, engine_info],
            pool_ids=[*compute_engine_pool_ids(args), compute_router_pool_id(0)],
        )
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
