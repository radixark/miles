import asyncio
from types import SimpleNamespace

import pytest

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_provider.base import CellInfo


def _make_cell_info(*, cell_id: str = "inference-engine-0-0-0", workers_hash: str = "pseudo-hash-0") -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        spec_name="inference-engine-0-0",
        worker_names=[f"{cell_id}-0"],
        workers_hash=workers_hash,
        meta=dict(
            model_id="default",
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


def _make_controller(servers: dict) -> InferenceController:
    controller = InferenceController.__new__(InferenceController)
    controller.args = SimpleNamespace(debug_train_only=False, use_fault_tolerance=False, ci_test=False, colocate=False)
    controller.servers = servers
    controller.rollout_engine_lock = None
    controller.context_lock = ContextLock("InferenceController")
    controller._health_checker_activeness = True
    return controller


class TestHealthCheckerActiveness:
    @pytest.mark.asyncio
    async def test_offload_pauses_probing_before_putting_engines_to_sleep(self):
        """A slept engine cannot answer /health_generate, so probing must stop first."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})

        await controller.offload()

        assert not controller._health_checker_activeness
        assert srv.calls == [("offload",)]

    @pytest.mark.asyncio
    async def test_starting_a_weight_update_pauses_probing(self):
        """Engines are unusable while their weights are being replaced."""
        controller = _make_controller({"default": _RecordingServer()})

        info = await controller.start_update_weights()
        await controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)

        assert not controller._health_checker_activeness

    @pytest.mark.asyncio
    async def test_preparing_a_rollout_resumes_probing(self):
        """Probing comes back exactly when the engines start serving traffic again."""
        controller = _make_controller({"default": _RecordingServer()})
        controller._health_checker_activeness = False

        await controller.prepare_rollout(rollout_id=0)

        assert controller._health_checker_activeness

    @pytest.mark.asyncio
    async def test_preparing_an_eval_resumes_probing(self):
        """Eval drives the same engines as a rollout does."""
        controller = _make_controller({"default": _RecordingServer()})
        controller._health_checker_activeness = False

        await controller.prepare_eval()

        assert controller._health_checker_activeness


class TestReconcile:
    @pytest.mark.asyncio
    async def test_an_observed_untracked_cell_is_added_to_its_model_server(self):
        """A newly observed engine cell lands in the server named by its model_id meta."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})
        info = _make_cell_info()

        await controller._reconcile(info.cell_id, info)

        assert srv.calls == [("add", info.cell_id)]

    @pytest.mark.asyncio
    async def test_a_disappeared_tracked_cell_is_removed(self):
        """A tracked cell reported as gone is removed even though no meta is observable."""
        info = _make_cell_info()
        srv = _RecordingServer({info.cell_id: SimpleNamespace(meta=_make_cell_meta(info))})
        controller = _make_controller({"default": srv})

        await controller._reconcile(info.cell_id, None)

        assert srv.calls == [("remove", info.cell_id)]
        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_workers_hash_change_replaces_the_cell(self):
        """A relaunched cell (new workers_hash) is removed then re-added, in that order."""
        old_info = _make_cell_info(workers_hash="pseudo-hash-0")
        srv = _RecordingServer({old_info.cell_id: SimpleNamespace(meta=_make_cell_meta(old_info))})
        controller = _make_controller({"default": srv})
        new_info = _make_cell_info(workers_hash="pseudo-hash-1")

        await controller._reconcile(new_info.cell_id, new_info)

        assert srv.calls == [("remove", new_info.cell_id), ("add", new_info.cell_id)]

    @pytest.mark.asyncio
    async def test_an_unchanged_tracked_cell_is_a_noop(self):
        """A tracked cell observed with the same workers_hash triggers no bookkeeping change."""
        info = _make_cell_info()
        srv = _RecordingServer({info.cell_id: SimpleNamespace(meta=_make_cell_meta(info))})
        controller = _make_controller({"default": srv})

        await controller._reconcile(info.cell_id, info)

        assert srv.calls == []

    @pytest.mark.asyncio
    async def test_a_disappeared_untracked_cell_is_a_noop(self):
        """A vanished cell that was never tracked (e.g. a router) triggers nothing."""
        srv = _RecordingServer()
        controller = _make_controller({"default": srv})

        await controller._reconcile("miles-router-0-0", None)

        assert srv.calls == []


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
