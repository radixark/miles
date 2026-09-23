import asyncio
import logging
from argparse import Namespace
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.server_cell import ServerCellMetadata
from miles.ray.train_actor import WeightUpdateOutput
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker


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
        self.is_errored = False

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
    def __init__(self, all_server_cells: dict[str, _ColocatedCellStub]) -> None:
        self.all_server_cells = all_server_cells
        self.health_checker_activeness = ActivenessTracker(active=True)

    @property
    def normal_server_cells(self) -> dict[str, _ColocatedCellStub]:
        return {cell_id: cell for cell_id, cell in self.all_server_cells.items() if not cell.is_errored}


def _make_inference_controller(**arg_overrides: object) -> InferenceController:
    return InferenceController(make_args(**arg_overrides), engine_provider=None, router_providers=[])


@pytest.mark.asyncio
async def test_controller_pauses_health_checks_before_snapshotting_the_engines():
    """``start_update_weights`` pauses the health monitor, then readies the cells, then reads the engine set."""
    order: list[str] = []
    controller = _make_inference_controller()

    async def _record_pause(model_id: str | None = None) -> None:
        order.append("health_monitoring_pause")

    async def _record_ensure_cells_ready(model_id: str | None = None) -> None:
        order.append("ensure_cells_ready")

    def _record_snapshot(model_id: str | None = None) -> None:
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

    def _record_snapshot(model_id: str | None = None) -> None:
        init_counts_at_snapshot.append(cell.init_count)
        return None

    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert cell.init_count == 1
    assert init_counts_at_snapshot == [1]


def _orchestration_args(**overrides) -> Namespace:
    values = dict(
        debug_train_only=False,
        debug_rollout_only=False,
        start_rollout_id=0,
        ci_ft_test_actions=None,
        ci_ft_test_actions_path=None,
        mini_ft_controller_enable=True,
        mini_ft_controller_poll_interval=0.01,
        log_inference_engine_weight_checksums=True,
    )
    values.update(overrides)
    return Namespace(**values)


def _actor_model(order: list[str]) -> MagicMock:
    async def _record_update_weights(*, info: object, rollout_id: int | None = None) -> int:
        order.append("trainer_update_weights")
        return 11

    actor_model = MagicMock()
    actor_model.update_weights = AsyncMock(side_effect=_record_update_weights)
    return actor_model


@pytest.mark.asyncio
async def test_the_script_brackets_the_broadcast_with_start_and_end_update_weights():
    """The fault-tolerant trainer runs the actual update RPC strictly inside the update window."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _OrderRecordingInferenceController(order)

    await update_weights(
        _orchestration_args(), _actor_model(order), MagicMock(set_weight_version=AsyncMock()), inference_controller
    )

    assert order[:3] == ["start_update_weights", "trainer_update_weights", "end_update_weights"]


@pytest.mark.asyncio
async def test_the_script_hands_end_update_weights_the_snapshot_start_returned():
    """The snapshot start_update_weights returned is handed back to end_update_weights unchanged."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _OrderRecordingInferenceController(order)

    await update_weights(
        _orchestration_args(), _actor_model(order), MagicMock(set_weight_version=AsyncMock()), inference_controller
    )

    _assert_the_snapshot_is_handed_back_unchanged(inference_controller)


@pytest.mark.asyncio
async def test_the_window_is_scoped_to_the_policy_the_script_is_publishing():
    """Without the scope, one policy's trainer broadcasts its weights into another policy's engines."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _OrderRecordingInferenceController(order)

    with patch("miles.ray.placement_group.is_event_logger_initialized", return_value=True), patch(
        "miles.ray.placement_group.get_event_logger"
    ), patch("miles.ray.placement_group.flatten_inference_engine_checksums", return_value=[]):
        await update_weights(
            _orchestration_args(),
            _actor_model(order),
            MagicMock(set_weight_version=AsyncMock()),
            inference_controller,
            rollout_id=3,
            trainer_model_id="alpha",
        )

    calls = {name: kwargs for name, _args, kwargs in inference_controller.calls}
    assert calls["start_update_weights"] == dict(model_id="alpha")
    assert calls["check_weights"] == dict(action="checksum", model_id="alpha")


def test_fsdp_updater_flushes_only_after_every_engine_is_paused():
    """Each weight-update phase finishes on every engine before the next phase starts on any."""
    from unittest.mock import patch

    from miles.backends.fsdp_utils.update_weight_utils import UpdateWeightFromTensor

    order: list[str] = []
    pause_modes: list[str] = []

    class _Client:
        def __init__(self, index: int):
            self._index = index

        async def pause_generation(self, mode: str = "retract"):
            order.append(f"pause-{self._index}")
            pause_modes.append(mode)

        async def flush_cache(self):
            order.append(f"flush-{self._index}")

        async def begin_weight_update(self, selector: str = "all"):
            order.append(f"begin-{self._index}")

        async def end_weight_update(self):
            order.append(f"end-{self._index}")

        async def continue_generation(self):
            order.append(f"continue-{self._index}")

    updater = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(update_weight_buffer_size=1 << 30)
    updater.weight_version = 0
    updater.model = MagicMock()
    updater.model.state_dict.return_value = {}
    updater.rollout_engines = [_Client(0), _Client(1)]

    module = "miles.backends.fsdp_utils.update_weight_utils"
    with patch(f"{module}.dist") as dist_mock, patch(f"{module}.get_gloo_group", return_value=MagicMock()):
        dist_mock.get_rank.return_value = 0
        updater.update_weights()

    assert set(order[:2]) == {"pause-0", "pause-1"}
    assert set(order[2:4]) == {"flush-0", "flush-1"}
    assert set(order[4:6]) == {"begin-0", "begin-1"}
    assert set(order[6:8]) == {"end-0", "end-1"}
    assert set(order[8:]) == {"continue-0", "continue-1"}
    assert pause_modes == ["retract", "retract"]


def _checksum_response(engine_checksums: list[dict[str, str]]) -> list:
    """Build a flat per-engine check_weights('checksum') response."""
    return [
        {
            "success": True,
            "message": "ok",
            "ranks": [{"checksums": cs, "parallelism_info": [{"role": "target", "rank": 0}]}],
        }
        for cs in engine_checksums
    ]


class TestTheScriptLogsTheChecksumsTheEnginesNowServe:
    @staticmethod
    async def _log(
        args: Namespace, *, response=None, initialized: bool = True, trainer_model_id: str | None = None
    ) -> tuple[MagicMock, MagicMock]:
        from miles.ray.placement_group import _maybe_log_inference_engine_weight_checksums

        inference_controller = MagicMock()
        inference_controller.check_weights = AsyncMock(return_value=response) if response is not None else MagicMock()
        event_logger = MagicMock()
        with patch("miles.ray.placement_group.is_event_logger_initialized", return_value=initialized), patch(
            "miles.ray.placement_group.get_event_logger", return_value=event_logger
        ):
            await _maybe_log_inference_engine_weight_checksums(
                args, inference_controller=inference_controller, rollout_id=0, trainer_model_id=trainer_model_id
            )
        return inference_controller, event_logger

    async def test_checksums_not_requested_does_not_call_check_weights(self):
        """Without the explicit opt-in, no check_weights request is issued even with an event logger."""
        inference_controller, _ = await self._log(
            _orchestration_args(log_inference_engine_weight_checksums=False), initialized=True
        )

        inference_controller.check_weights.assert_not_called()

    async def test_no_event_logger_does_not_call_check_weights(self):
        """Without an initialized event logger, no check_weights request is issued."""
        inference_controller, _ = await self._log(_orchestration_args(), initialized=False)

        inference_controller.check_weights.assert_not_called()

    async def test_debug_train_only_skips_collection(self):
        """Without real rollout engines (debug_train_only), no check_weights request is issued."""
        inference_controller, _ = await self._log(_orchestration_args(debug_train_only=True))

        inference_controller.check_weights.assert_not_called()

    async def test_debug_rollout_only_skips_collection(self):
        """Without real train engines pushing weights (debug_rollout_only), no check_weights request is issued."""
        inference_controller, _ = await self._log(_orchestration_args(debug_rollout_only=True))

        inference_controller.check_weights.assert_not_called()

    async def test_enabled_logs_one_event_per_rollout(self):
        """With event logger on and real engines, one event holds every engine's checksums."""
        response = _checksum_response([{"w": "e0"}, {"w": "e1"}])

        inference_controller, event_logger = await self._log(_orchestration_args(), response=response)

        inference_controller.check_weights.assert_awaited_once_with(action="checksum", model_id=None)
        event_logger.log.assert_called_once()
        assert event_logger.log.call_args.args[1] == dict(
            rollout_id=0, trainer_model_id=None, engine_checksums=[{"rank0/w": "e0"}, {"rank0/w": "e1"}]
        )

    async def test_a_named_policy_stamps_its_own_id_on_the_event(self):
        """A multi policy run's event names the policy, so the comparator can tell two policies apart."""
        response = _checksum_response([{"w": "e0"}])

        inference_controller, event_logger = await self._log(
            _orchestration_args(), response=response, trainer_model_id="solver"
        )

        inference_controller.check_weights.assert_awaited_once_with(action="checksum", model_id="solver")
        assert event_logger.log.call_args.args[1] == dict(
            rollout_id=0, trainer_model_id="solver", engine_checksums=[{"rank0/w": "e0"}]
        )


class TestTheChecksumRecordKeepsOnlyThisPublication:
    @pytest.fixture
    def event_log_dir(self, tmp_path: Path) -> Iterator[Path]:
        set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="main")))
        try:
            yield tmp_path
        finally:
            set_event_logger(None)

    @staticmethod
    async def _log(
        *,
        check_weights: Callable[..., Awaitable[Any]],
        snapshot: dict[str, str],
        failed: tuple[str, ...] = (),
        weight_version: int | None = 11,
        **arg_overrides: object,
    ) -> None:
        from miles.ray.placement_group import _maybe_log_inference_engine_weight_checksums

        await _maybe_log_inference_engine_weight_checksums(
            _orchestration_args(**arg_overrides),
            inference_controller=SimpleNamespace(check_weights=check_weights),
            rollout_id=0,
            trainer_model_id=None,
            output=WeightUpdateOutput(
                weight_version=weight_version,
                failed_cell_ids=failed,
                debug_trainer_load_state_timestamp=1.0,
                debug_weight_update_id="update-11",
            ),
            snapshot_cell_id_to_hashes=snapshot,
        )

    @staticmethod
    def _answering(response: list[tuple[ServerCellMetadata, dict[str, Any]]]) -> Callable[..., Awaitable[Any]]:
        async def _check_weights(**_kwargs: object) -> list[tuple[ServerCellMetadata, dict[str, Any]]]:
            return response

        return _check_weights

    @staticmethod
    def _recorded(log_dir: Path) -> list[InferenceEngineWeightChecksumEvent]:
        return [e for e in read_events(log_dir) if isinstance(e, InferenceEngineWeightChecksumEvent)]

    async def test_a_cell_the_update_failed_on_is_left_out(self, event_log_dir: Path) -> None:
        """A failed cell still serves the previous version, so its checksum would be charged to this one."""
        response = _checksum_response([{"w": "new"}, {"w": "old"}])

        await self._log(
            check_weights=self._answering(response),
            snapshot={"cell-0": "incarnation-0", "cell-1": "incarnation-1"},
            failed=("cell-1",),
        )

        [event] = self._recorded(event_log_dir)
        assert [snapshot.cell_id for snapshot in event.engine_snapshots] == ["cell-0"]

    async def test_a_cell_replaced_since_the_snapshot_is_left_out(self, event_log_dir: Path) -> None:
        """A same-named cell with a new incarnation never received this update."""
        response = _checksum_response([{"w": "new"}, {"w": "fresh"}])

        await self._log(
            check_weights=self._answering(response),
            snapshot={"cell-0": "incarnation-0", "cell-1": "incarnation-before-restart"},
        )

        [event] = self._recorded(event_log_dir)
        assert [(s.cell_id, s.workers_hash) for s in event.engine_snapshots] == [("cell-0", "incarnation-0")]

    async def test_a_cell_outside_the_snapshot_is_left_out(self, event_log_dir: Path) -> None:
        """A cell that joined after the window opened was never a target of this update."""
        response = _checksum_response([{"w": "new"}, {"w": "joined-late"}])

        await self._log(check_weights=self._answering(response), snapshot={"cell-0": "incarnation-0"})

        [event] = self._recorded(event_log_dir)
        assert [snapshot.cell_id for snapshot in event.engine_snapshots] == ["cell-0"]

    async def test_an_unpublished_update_asks_no_engine(self, event_log_dir: Path) -> None:
        """Without a published version there is no version the checksums could be attributed to."""
        asked: list[dict[str, object]] = []

        async def _check_weights(**kwargs: object) -> list[tuple[ServerCellMetadata, dict[str, Any]]]:
            asked.append(kwargs)
            return _checksum_response([{"w": "x"}])

        await self._log(check_weights=_check_weights, snapshot={"cell-0": "incarnation-0"}, weight_version=None)

        assert asked == []
        assert self._recorded(event_log_dir) == []

    async def test_a_hanging_engine_is_deadlined_without_failing_the_update(self, event_log_dir: Path) -> None:
        """Evidence collection is best effort, so a stuck engine must not block or fail publication."""

        async def _hang(**_kwargs: object) -> None:
            await asyncio.Event().wait()

        await self._log(
            check_weights=_hang, snapshot={"cell-0": "incarnation-0"}, update_weight_engine_request_timeout=0.01
        )

        assert self._recorded(event_log_dir) == []

    async def test_a_failed_engine_body_is_logged_instead_of_raised(
        self, event_log_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed check_weights answer must not turn a published update into a failed one."""
        [(meta, body)] = _checksum_response([{"w": "x"}])

        with caplog.at_level(logging.ERROR, logger="miles.ray.placement_group"):
            await self._log(
                check_weights=self._answering([(meta, {**body, "success": False})]),
                snapshot={"cell-0": "incarnation-0"},
            )

        assert self._recorded(event_log_dir) == []
        assert "Could not record inference engine checksum observation" in caplog.text
