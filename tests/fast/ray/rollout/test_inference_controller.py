import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout import inference_controller as inference_controller_module
from miles.ray.rollout.inference_controller import InferenceController, _compute_server_cell_meta_from_info
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCellMetadata
from miles.ray.specs.inference import compute_engine_pool_ids, compute_router_pool_id, specs_inference_engine
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.workers.worker_provider.base import CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import WorkerMetaContext


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
    def __init__(self, server_cells: dict | None = None, *, model_name: str = "model", update_weights: bool = False):
        self.server_cells = server_cells or {}
        self.update_weights = update_weights
        self.model_name = model_name
        self.calls: list[tuple] = []
        self.api_clients: list = []
        self.engine_gpu_counts: list[int] = []
        self.engine_gpu_offsets: list[int] = []

    async def offload(self, tags=None):
        self.calls.append(("offload",))

    async def check_weights(self, action, allow_quant_error=False, selector="all", skip_list=None):
        self.calls.append(("check_weights", action))
        return [self.model_name]

    async def add_cell(self, cell_meta: ServerCellMetadata):
        self.calls.append(("add", cell_meta.cell_id))
        self.server_cells[cell_meta.cell_id] = SimpleNamespace(meta=cell_meta)

    async def remove_cell(self, cell_id: str):
        self.calls.append(("remove", cell_id))
        del self.server_cells[cell_id]

    async def wait_expected_num_cells(self) -> None:
        return None


def _make_controller(servers: dict) -> InferenceController:
    controller = InferenceController.__new__(InferenceController)
    controller.args = SimpleNamespace(
        debug_train_only=False,
        use_fault_tolerance=False,
        ci_test=False,
        colocate=False,
        offload_rollout_level=["kv_cache", "weight"],
    )
    controller.servers = servers
    controller.context_lock = ContextLock("InferenceController")
    controller._health_checker_activeness = ActivenessTracker(active=True)
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


class _FakeWorkerProvider:
    def __init__(self, cell_infos: list[CellInfo]) -> None:
        self._cell_infos = cell_infos
        self._pools: list[str] = []
        self.watched_pool_ids: list[str] | None = None

    def created_with(self, pool_ids: list[str]) -> "_FakeWorkerProvider":
        self._pools = list(pool_ids)
        return self

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        pool_ids = self._pools
        self.watched_pool_ids = list(pool_ids)
        for info in self._cell_infos:
            if info.pool_id in pool_ids:
                await reconcile(info.cell_id, info)

        async def _stop_watch() -> None:
            return None

        return _stop_watch


def _patch_init(
    monkeypatch: pytest.MonkeyPatch, *, provider: _FakeWorkerProvider, servers: dict[str, _RecordingServer]
) -> None:
    async def _fake_create_rollout_servers(args: Namespace, **kwargs: Any) -> dict[str, _RecordingServer]:
        return servers

    monkeypatch.setattr(inference_controller_module, "create_rollout_servers", _fake_create_rollout_servers)
    monkeypatch.setattr(
        inference_controller_module,
        "RayWorkerProvider",
        SimpleNamespace(create=lambda *, pool_ids: provider.created_with(pool_ids)),
        raising=True,
    )


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
    async def test_init_watches_exactly_the_engine_specs(self, monkeypatch: pytest.MonkeyPatch):
        """init must subscribe to engine specs only; a router spec here is reconciled as an engine."""
        args = make_args()
        provider = _FakeWorkerProvider([])
        _patch_init(monkeypatch, provider=provider, servers={"default": _RecordingServer()})

        await InferenceController(args).init()

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
        provider = _FakeWorkerProvider([router_info, engine_info])
        srv = _RecordingServer()
        _patch_init(monkeypatch, provider=provider, servers={"default": srv})

        await InferenceController(args).init()

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
        srv = RolloutServer(server_cells={}, args=SimpleNamespace(), context_lock=controller.context_lock)
        controller.servers = {"default": srv}
        info = _make_cell_info()

        await controller._reconcile(info.cell_id, None)
        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_server_holding_a_foreign_lock_is_rejected(self):
        """A server wired up with its own lock instead of the controller's is a wiring bug."""
        controller = _make_controller({})
        srv = RolloutServer(server_cells={}, args=SimpleNamespace(), context_lock=ContextLock("InferenceController"))
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
