from types import SimpleNamespace

import pytest

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.server_cell import ServerCellMetadata
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
        self.calls: list[tuple] = []

    async def add_cell(self, cell_meta: ServerCellMetadata):
        self.calls.append(("add", cell_meta.cell_id))
        self.server_cells[cell_meta.cell_id] = SimpleNamespace(meta=cell_meta)

    async def remove_cell(self, cell_id: str):
        self.calls.append(("remove", cell_id))
        del self.server_cells[cell_id]


def _make_controller(servers: dict) -> InferenceController:
    controller = InferenceController.__new__(InferenceController)
    controller.servers = servers
    return controller


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
