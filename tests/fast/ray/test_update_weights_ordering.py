from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock


class _OrderRecordingInferenceController:
    def __init__(self, order: list[str]):
        self._order = order
        self.calls: list[tuple[str, tuple, dict]] = []
        self.results: dict[str, MagicMock] = {}

    def __getattr__(self, name: str):
        recorder = self

        async def _method(*args, **kwargs):
            recorder._order.append(name)
            recorder.calls.append((name, args, kwargs))
            result = MagicMock()
            recorder.results[name] = result
            return result

        return _method


def _assert_the_snapshot_is_handed_back_unchanged(controller: _OrderRecordingInferenceController) -> None:
    end_kwargs: list[dict] = [kwargs for name, _args, kwargs in controller.calls if name == "end_update_weights"]

    assert len(end_kwargs) == 1
    assert (
        end_kwargs[0]["snapshot_cell_id_to_hashes"]
        is controller.results["start_update_weights"].snapshot_cell_id_to_hashes
    )


class _ColocatedCellStub:
    def __init__(self) -> None:
        self.init_count = 0
        self.ready = False

    async def init(self) -> None:
        self.init_count += 1
        self.ready = True

    @property
    def is_uninitialized(self) -> bool:
        return not self.ready

    @property
    def is_pending_weights_or_serving(self) -> bool:
        return self.ready


class _ServerStub:
    def __init__(self, server_cells: dict[str, _ColocatedCellStub]) -> None:
        self.server_cells = server_cells


def _make_inference_controller(**arg_overrides: object) -> InferenceController:
    return InferenceController(make_args(**arg_overrides))


@pytest.mark.asyncio
async def test_controller_pauses_health_checks_before_snapshotting_the_engines():
    """``start_update_weights`` pauses the health monitor, then readies the cells, then reads the engine set."""
    order: list[str] = []
    controller = _make_inference_controller()

    async def _record_pause() -> None:
        order.append("health_monitoring_pause")

    async def _record_ensure_cells_ready() -> None:
        order.append("ensure_cells_ready")

    def _record_snapshot() -> None:
        order.append("get_updatable_server")
        return None

    controller.context_lock = ContextLock("InferenceController")
    controller.args = Namespace(colocate=False)
    controller.servers = {}
    controller._health_monitoring_pause = _record_pause
    controller._ensure_cells_ready = _record_ensure_cells_ready
    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert order == ["health_monitoring_pause", "ensure_cells_ready", "get_updatable_server"]


@pytest.mark.asyncio
async def test_start_update_weights_initializes_colocated_cells_before_snapshotting_the_engines():
    """A colocated cell is initialized inside the weight update window, before the engine snapshot is taken."""
    controller = _make_inference_controller(colocate=True)
    cell = _ColocatedCellStub()
    controller.servers = {"default": _ServerStub({"a": cell})}
    init_counts_at_snapshot: list[int] = []

    def _record_snapshot() -> None:
        init_counts_at_snapshot.append(cell.init_count)
        return None

    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert cell.init_count == 1
    assert init_counts_at_snapshot == [1]


def _make_controller(order: list[str]):
    from miles.ray.train.group import TrainerController

    group = TrainerController.__new__(TrainerController)
    group.args = Namespace(debug_train_only=False, debug_rollout_only=False)
    group._inference_controller = _OrderRecordingInferenceController(order)

    async def _record_execute_first_alive(*args: object, **kwargs: object) -> list[int]:
        order.append("execute_first_alive")
        return [1]

    group._execute_first_alive = AsyncMock(side_effect=_record_execute_first_alive)
    group._maybe_log_inference_engine_weight_checksums = AsyncMock()
    return group


@pytest.mark.asyncio
async def test_the_trainer_brackets_the_broadcast_with_start_and_end_update_weights():
    """The fault-tolerant trainer runs the actual update RPC strictly inside the update window."""
    order: list[str] = []
    group = _make_controller(order)

    await group.update_weights()

    assert order == ["start_update_weights", "execute_first_alive", "end_update_weights"]
    group._execute_first_alive.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_trainer_hands_end_update_weights_the_snapshot_start_returned():
    """The snapshot start_update_weights returned is handed back to end_update_weights unchanged."""
    order: list[str] = []
    group = _make_controller(order)

    await group.update_weights()

    _assert_the_snapshot_is_handed_back_unchanged(group._inference_controller)
