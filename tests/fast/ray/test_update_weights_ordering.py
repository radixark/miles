from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from miles.ray.actor_group import RayTrainGroup
from miles.ray.rollout.inference_controller import InferenceController


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


@pytest.mark.asyncio
async def test_controller_pauses_health_checks_before_snapshotting_the_engines():
    """``start_update_weights`` pauses the health monitor before it reads the engine set."""
    order: list[str] = []
    controller = InferenceController.__new__(InferenceController)

    async def _record_pause() -> None:
        order.append("health_monitoring_pause")

    def _record_snapshot():
        order.append("get_updatable_server")
        return None

    controller._health_monitoring_pause = _record_pause
    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert order == ["health_monitoring_pause", "get_updatable_server"]


def _make_v1_group(order: list[str]) -> RayTrainGroup:
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = Namespace(debug_train_only=False, debug_rollout_only=False, use_fault_tolerance=False)
    group._inference_controller = _OrderRecordingInferenceController(order)

    async def _record_broadcast(*args: object, **kwargs: object) -> None:
        order.append("broadcast")

    group._broadcast = AsyncMock(side_effect=_record_broadcast)
    return group


def _make_v2_group(order: list[str]):
    from miles.ray.train.group import TrainerController as FaultTolerantTrainGroup

    group = TrainerController.__new__(TrainerController)
    group.args = Namespace(debug_train_only=False, debug_rollout_only=False)
    group._inference_controller = _OrderRecordingInferenceController(order)

    async def _record_execute_first_alive(*args: object, **kwargs: object) -> None:
        order.append("execute_first_alive")

    group._execute_first_alive = AsyncMock(side_effect=_record_execute_first_alive)
    group._maybe_log_inference_engine_weight_checksums = AsyncMock()
    return group


@pytest.mark.asyncio
async def test_v1_brackets_the_broadcast_with_start_and_end_update_weights():
    """The trainer broadcast is recorded strictly between the start and end of the update window."""
    order: list[str] = []
    group = _make_v1_group(order)

    await group.update_weights()

    assert order == ["start_update_weights", "broadcast", "end_update_weights"]
    group._broadcast.assert_awaited_once()


@pytest.mark.asyncio
async def test_v2_brackets_the_broadcast_with_start_and_end_update_weights():
    """The fault-tolerant trainer runs the actual update RPC strictly inside the update window."""
    order: list[str] = []
    group = _make_v2_group(order)

    await group.update_weights()

    assert order == ["start_update_weights", "execute_first_alive", "end_update_weights"]
    group._execute_first_alive.assert_awaited_once()


@pytest.mark.asyncio
async def test_v1_hands_end_update_weights_the_snapshot_start_returned():
    """A dropped or substituted snapshot leaves every pending cell unregistered with the router."""
    order: list[str] = []
    group = _make_v1_group(order)

    await group.update_weights()

    _assert_the_snapshot_is_handed_back_unchanged(group._inference_controller)


@pytest.mark.asyncio
async def test_v2_hands_end_update_weights_the_snapshot_start_returned():
    """Same snapshot pass-through requirement on the fault-tolerant trainer group."""
    order: list[str] = []
    group = _make_v2_group(order)

    await group.update_weights()

    _assert_the_snapshot_is_handed_back_unchanged(group._inference_controller)


def test_fsdp_updater_flushes_only_after_every_engine_is_paused():
    """Every engine is paused before any engine is flushed."""
    from unittest.mock import patch

    from miles.backends.experimental.fsdp_utils.update_weight_utils import UpdateWeightFromTensor

    order: list[str] = []
    pause_modes: list[str] = []

    class _Client:
        def __init__(self, index: int):
            self._index = index

        async def pause_generation(self, *, mode: str = "retract"):
            order.append(f"pause-{self._index}")
            pause_modes.append(mode)

        async def flush_cache(self):
            order.append(f"flush-{self._index}")

        async def continue_generation(self):
            order.append(f"continue-{self._index}")

    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(update_weight_buffer_size=1 << 30)
    updater.weight_version = 0
    updater.model = MagicMock()
    updater.model.state_dict.return_value = {}
    updater.rollout_engines = [_Client(0), _Client(1)]

    module = "miles.backends.experimental.fsdp_utils.update_weight_utils"
    with patch(f"{module}.dist") as dist_mock, patch(f"{module}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater.update_weights()

    assert set(order[:2]) == {"pause-0", "pause-1"}
    assert set(order[2:4]) == {"flush-0", "flush-1"}
    assert set(order[4:]) == {"continue-0", "continue-1"}
    assert pause_modes == ["retract", "retract"]
