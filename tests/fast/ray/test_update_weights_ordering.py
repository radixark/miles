import asyncio
import contextlib
from argparse import Namespace
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController, UpdatableEngines
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.workers.rpc.client import handle as rpc_handle_module
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.server.app import create_rpc_app


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


class _RpcWeightUpdateController:
    def __init__(self, *, block_start: bool = False) -> None:
        self.start_called = asyncio.Event()
        self.start_release = asyncio.Event()
        self.end_snapshots: list[dict[str, str]] = []
        self.abort_snapshots: list[dict[str, str]] = []
        if not block_start:
            self.start_release.set()

    async def start_update_weights(self, model_id: str | None = None) -> UpdatableEngines:
        self.start_called.set()
        await self.start_release.wait()
        return UpdatableEngines(
            rollout_engines=[],
            engine_gpu_counts=[],
            engine_gpu_offsets=[],
            snapshot_cell_id_to_hashes={"cell": "generation"},
        )

    async def end_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        self.end_snapshots.append(snapshot_cell_id_to_hashes)

    async def abort_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        self.abort_snapshots.append(snapshot_cell_id_to_hashes)


class _RpcWeightUpdateTrainer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def update_weights(self, info: UpdatableEngines, rollout_id: int | None = None) -> int:
        self.started.set()
        await self.release.wait()
        self.finished.set()
        return 11


class _RetryWindowController:
    def __init__(
        self,
        order: list[str],
        *,
        end_errors: list[Exception] | None = None,
        abort_errors: list[Exception] | None = None,
        wait_for_heal: bool = False,
    ) -> None:
        self._order = order
        self._end_errors = list(end_errors or [])
        self._abort_errors = list(abort_errors or [])
        self._generation = 0
        self._wait_for_heal = wait_for_heal
        self.context_lock = ContextLock("InferenceController")
        self.heal_finished = asyncio.Event()

    async def start_update_weights(self, model_id: str | None = None) -> UpdatableEngines:
        await self.context_lock.acquire()
        self.context_lock.detach()
        self._generation += 1
        generation = f"generation-{self._generation}"
        self._order.append(f"start:{generation}")
        return UpdatableEngines(
            rollout_engines=[],
            engine_gpu_counts=[],
            engine_gpu_offsets=[],
            snapshot_cell_id_to_hashes={"cell": generation},
        )

    async def end_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        self._order.append(f"end:{snapshot_cell_id_to_hashes}")
        if self._end_errors:
            error = self._end_errors.pop(0)
            self.context_lock.reattach()
            self.context_lock.release()
            raise error
        self.context_lock.reattach()
        self.context_lock.release()

    async def abort_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        self._order.append(f"abort:{snapshot_cell_id_to_hashes}")
        if self._abort_errors:
            error = self._abort_errors.pop(0)
            self.context_lock.reattach()
            self.context_lock.release()
            raise error
        self.context_lock.reattach()
        self.context_lock.release()

    async def heal_dead_cell(self) -> None:
        async with self.context_lock:
            self._order.append("heal")
            self.heal_finished.set()

    async def wait_expected_num_cells(self) -> None:
        self._order.append("wait:rollout")
        if self._wait_for_heal:
            await self.heal_finished.wait()

    async def wait_idle(self, *, timeout: float) -> None:
        self._order.append("wait_idle:inference")


@contextlib.asynccontextmanager
async def _rpc_handle(worker: object, *, call_timeout_seconds: float = 3600.0) -> AsyncIterator[RpcWorkerHandle]:
    app = create_rpc_app(worker)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
            yield RpcWorkerHandle(
                type(worker),
                server_url="http://testserver",
                http_client=client,
                call_timeout_seconds=call_timeout_seconds,
            )


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
        self.health_checker_activeness = ActivenessTracker(active=True)


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
        save_inference_engine_weight_checksum=True,
        ci_ft_test_actions=None,
        ci_ft_test_actions_path=None,
        mini_ft_controller_enable=True,
        mini_ft_controller_poll_interval=0.01,
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
async def test_cancelling_the_broadcast_waits_for_completion_before_closing_the_update_window():
    """Cancelling a policy waits for its in-flight broadcast before accepting the completed weights."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    broadcast_started = asyncio.Event()
    hold_broadcast = asyncio.Event()
    inference_controller = _OrderRecordingInferenceController(order)

    async def _block_update_weights(*, info: object, rollout_id: int | None = None) -> int:
        order.append("trainer_update_weights")
        broadcast_started.set()
        await hold_broadcast.wait()
        return 11

    actor_model = MagicMock(update_weights=AsyncMock(side_effect=_block_update_weights))
    task = asyncio.create_task(
        update_weights(
            _orchestration_args(),
            actor_model,
            MagicMock(set_weight_version=AsyncMock()),
            inference_controller,
        )
    )
    await asyncio.wait_for(broadcast_started.wait(), timeout=5)

    task.cancel()
    await asyncio.sleep(0)
    task_was_done_while_broadcast_was_blocked = task.done()
    calls_while_broadcast_was_blocked = [name for name, _args, _kwargs in inference_controller.calls]

    hold_broadcast.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not task_was_done_while_broadcast_was_blocked
    assert calls_while_broadcast_was_blocked == ["start_update_weights"]
    assert order == ["start_update_weights", "trainer_update_weights", "end_update_weights"]
    _assert_the_snapshot_is_handed_back_unchanged(inference_controller)


@pytest.mark.asyncio
async def test_cancelling_an_in_flight_rpc_broadcast_drains_it_before_releasing_the_window():
    """A caller cancellation cannot overtake the remote broadcast it already submitted."""
    from miles.ray.placement_group import update_weights

    controller = _RpcWeightUpdateController()
    trainer = _RpcWeightUpdateTrainer()
    async with _rpc_handle(controller) as controller_handle, _rpc_handle(trainer) as trainer_handle:
        task = asyncio.create_task(
            update_weights(
                _orchestration_args(),
                trainer_handle,
                MagicMock(set_weight_version=AsyncMock()),
                controller_handle,
            )
        )
        await asyncio.wait_for(trainer.started.wait(), timeout=5)

        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task_was_done_while_remote_was_blocked = task.done()
        snapshots_before_remote_finished = list(controller.end_snapshots)

        trainer.release.set()
        await asyncio.wait_for(trainer.finished.wait(), timeout=5)
        with pytest.raises(asyncio.CancelledError):
            await task

    assert not task_was_done_while_remote_was_blocked
    assert snapshots_before_remote_finished == []
    assert controller.end_snapshots == [{"cell": "generation"}]


@pytest.mark.asyncio
async def test_cancelling_an_in_flight_rpc_start_still_closes_the_window_it_opens():
    """A start RPC abandoned by its caller cannot later detach an update window nobody closes."""
    from miles.ray.placement_group import update_weights

    controller = _RpcWeightUpdateController(block_start=True)
    trainer = _RpcWeightUpdateTrainer()
    trainer.release.set()
    async with _rpc_handle(controller) as controller_handle, _rpc_handle(trainer) as trainer_handle:
        task = asyncio.create_task(
            update_weights(
                _orchestration_args(),
                trainer_handle,
                MagicMock(set_weight_version=AsyncMock()),
                controller_handle,
            )
        )
        await asyncio.wait_for(controller.start_called.wait(), timeout=5)

        task.cancel()
        await asyncio.sleep(0)
        task_was_done_while_start_was_blocked = task.done()

        controller.start_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert not task_was_done_while_start_was_blocked
    assert controller.end_snapshots == [{"cell": "generation"}]


@pytest.mark.asyncio
async def test_a_timed_out_rpc_broadcast_drains_before_replacing_the_participants(monkeypatch: pytest.MonkeyPatch):
    """A client polling timeout cannot replace engines while its remote broadcast is still running."""
    from miles.ray import placement_group

    monkeypatch.setattr(rpc_handle_module, "_IDLE_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(placement_group, "_WEIGHT_UPDATE_RETRY_MAX_ATTEMPTS", 1)
    controller = _RpcWeightUpdateController()
    trainer = _RpcWeightUpdateTrainer()
    async with _rpc_handle(controller) as controller_handle, _rpc_handle(
        trainer, call_timeout_seconds=0.05
    ) as trainer_handle:
        task = asyncio.create_task(
            placement_group.update_weights(
                _orchestration_args(),
                trainer_handle,
                MagicMock(set_weight_version=AsyncMock()),
                controller_handle,
            )
        )
        await asyncio.wait_for(trainer.started.wait(), timeout=5)
        await asyncio.sleep(0.1)
        snapshots_while_remote_was_blocked = list(controller.abort_snapshots)

        trainer.release.set()
        await asyncio.wait_for(trainer.finished.wait(), timeout=5)
        with pytest.raises(TimeoutError, match="still pending"):
            await task

    assert snapshots_while_remote_was_blocked == []
    assert controller.abort_snapshots == [{"cell": "generation"}]
    assert controller.end_snapshots == []


@pytest.mark.asyncio
async def test_a_failed_broadcast_replaces_the_snapshot_without_readying_cells(monkeypatch: pytest.MonkeyPatch):
    """A failed policy broadcast replaces its engines without accepting partial weights."""
    from miles.ray import placement_group

    monkeypatch.setattr(placement_group, "_WEIGHT_UPDATE_RETRY_MAX_ATTEMPTS", 1)
    order: list[str] = []
    inference_controller = _OrderRecordingInferenceController(order)

    async def _fail_update_weights(*, info: object, rollout_id: int | None = None) -> int:
        order.append("trainer_update_weights")
        raise RuntimeError("broadcast failed")

    actor_model = MagicMock(update_weights=AsyncMock(side_effect=_fail_update_weights))

    with pytest.raises(RuntimeError, match="broadcast failed"):
        await placement_group.update_weights(
            _orchestration_args(),
            actor_model,
            MagicMock(set_weight_version=AsyncMock()),
            inference_controller,
        )

    assert order == ["start_update_weights", "trainer_update_weights", "abort_update_weights"]
    [abort_kwargs] = [kwargs for name, _args, kwargs in inference_controller.calls if name == "abort_update_weights"]
    assert (
        abort_kwargs["snapshot_cell_id_to_hashes"]
        is inference_controller.results["start_update_weights"].snapshot_cell_id_to_hashes
    )


@pytest.mark.asyncio
async def test_a_failed_broadcast_replaces_every_snapshot_participant_before_retrying():
    """A partial broadcast cannot leave any participant serving a mixture of old and new weights."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _RetryWindowController(order)
    broadcast_count = 0

    async def _broadcast(*, info: UpdatableEngines, rollout_id: int | None = None) -> int:
        nonlocal broadcast_count
        broadcast_count += 1
        [generation] = info.snapshot_cell_id_to_hashes.values()
        order.append(f"broadcast:{generation}")
        if broadcast_count == 1:
            raise RuntimeError("one engine died after its peers accepted partial weights")
        return 11

    await update_weights(
        _orchestration_args(),
        MagicMock(
            update_weights=AsyncMock(side_effect=_broadcast),
            wait_until_update_weights_ready=AsyncMock(side_effect=lambda: order.append("wait:trainer")),
        ),
        MagicMock(set_weight_version=AsyncMock()),
        inference_controller,
    )

    assert order == [
        "start:generation-1",
        "broadcast:generation-1",
        "abort:{'cell': 'generation-1'}",
        "wait:trainer",
        "wait:rollout",
        "start:generation-2",
        "broadcast:generation-2",
        "end:{'cell': 'generation-2'}",
    ]


@pytest.mark.asyncio
async def test_an_ambiguous_participant_replacement_never_starts_a_second_window():
    """A retry is unsafe until every process that may hold partial weights is confirmed stopped."""
    from miles.ray import placement_group
    from miles.utils.retry_utils import NonRetryableError

    order: list[str] = []
    inference_controller = _RetryWindowController(
        order,
        abort_errors=[RuntimeError("could not confirm old worker exit")],
    )

    with pytest.raises(NonRetryableError, match="did not close cleanly"):
        await placement_group.update_weights(
            _orchestration_args(),
            MagicMock(update_weights=AsyncMock(side_effect=RuntimeError("partial broadcast"))),
            MagicMock(set_weight_version=AsyncMock()),
            inference_controller,
        )

    assert order == [
        "start:generation-1",
        "abort:{'cell': 'generation-1'}",
    ]


@pytest.mark.asyncio
async def test_a_failed_broadcast_releases_the_window_before_retrying_with_a_fresh_snapshot():
    """A retry lets healing take the lock and snapshots the replacement engines from scratch."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _RetryWindowController(order, wait_for_heal=True)
    heal_task: asyncio.Task[None] | None = None
    broadcast_count = 0

    async def _broadcast(*, info: UpdatableEngines, rollout_id: int | None = None) -> int:
        nonlocal broadcast_count, heal_task
        broadcast_count += 1
        [generation] = info.snapshot_cell_id_to_hashes.values()
        order.append(f"broadcast:{generation}")
        if broadcast_count == 1:
            heal_task = asyncio.create_task(inference_controller.heal_dead_cell())
            await asyncio.sleep(0)
            assert not inference_controller.heal_finished.is_set()
            raise RuntimeError("dead rollout engine")
        assert inference_controller.heal_finished.is_set()
        return 11

    async def _wait_trainer() -> None:
        order.append("wait:trainer")

    actor_model = MagicMock(
        update_weights=AsyncMock(side_effect=_broadcast),
        wait_until_update_weights_ready=AsyncMock(side_effect=_wait_trainer),
    )

    await update_weights(
        _orchestration_args(),
        actor_model,
        MagicMock(set_weight_version=AsyncMock()),
        inference_controller,
    )
    assert heal_task is not None
    await asyncio.wait_for(heal_task, timeout=5)

    assert order == [
        "start:generation-1",
        "broadcast:generation-1",
        "abort:{'cell': 'generation-1'}",
        "heal",
        "wait:trainer",
        "wait:rollout",
        "start:generation-2",
        "broadcast:generation-2",
        "end:{'cell': 'generation-2'}",
    ]


@pytest.mark.asyncio
async def test_cancelling_a_retried_broadcast_drains_it_before_closing_its_fresh_window():
    """Cancellation during a retry cannot close its fresh window ahead of the in-flight broadcast."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _RetryWindowController(order)
    retry_started = asyncio.Event()
    retry_release = asyncio.Event()
    broadcast_count = 0

    async def _broadcast(*, info: UpdatableEngines, rollout_id: int | None = None) -> int:
        nonlocal broadcast_count
        broadcast_count += 1
        [generation] = info.snapshot_cell_id_to_hashes.values()
        order.append(f"broadcast:{generation}")
        if broadcast_count == 1:
            raise RuntimeError("dead rollout engine")
        retry_started.set()
        await retry_release.wait()
        return 11

    task = asyncio.create_task(
        update_weights(
            _orchestration_args(),
            MagicMock(
                update_weights=AsyncMock(side_effect=_broadcast),
                wait_until_update_weights_ready=AsyncMock(side_effect=lambda: order.append("wait:trainer")),
            ),
            MagicMock(set_weight_version=AsyncMock()),
            inference_controller,
        )
    )
    await asyncio.wait_for(retry_started.wait(), timeout=5)

    task.cancel()
    await asyncio.sleep(0)
    order_while_retry_was_blocked = list(order)

    retry_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert order_while_retry_was_blocked == [
        "start:generation-1",
        "broadcast:generation-1",
        "abort:{'cell': 'generation-1'}",
        "wait:trainer",
        "wait:rollout",
        "start:generation-2",
        "broadcast:generation-2",
    ]
    assert order[-1] == "end:{'cell': 'generation-2'}"


@pytest.mark.asyncio
async def test_a_second_failed_broadcast_closes_its_fresh_window_before_giving_up(monkeypatch: pytest.MonkeyPatch):
    """Exhausting broadcast attempts closes every fresh window without accepting either snapshot."""
    from miles.ray import placement_group

    monkeypatch.setattr(placement_group, "_WEIGHT_UPDATE_RETRY_MAX_ATTEMPTS", 2)
    order: list[str] = []
    inference_controller = _RetryWindowController(order)

    async def _broadcast(*, info: UpdatableEngines, rollout_id: int | None = None) -> int:
        [generation] = info.snapshot_cell_id_to_hashes.values()
        order.append(f"broadcast:{generation}")
        raise RuntimeError(generation)

    with pytest.raises(RuntimeError, match="generation-2"):
        await placement_group.update_weights(
            _orchestration_args(),
            MagicMock(
                update_weights=AsyncMock(side_effect=_broadcast),
                wait_until_update_weights_ready=AsyncMock(side_effect=lambda: order.append("wait:trainer")),
            ),
            MagicMock(set_weight_version=AsyncMock()),
            inference_controller,
        )

    assert order == [
        "start:generation-1",
        "broadcast:generation-1",
        "abort:{'cell': 'generation-1'}",
        "wait:trainer",
        "wait:rollout",
        "start:generation-2",
        "broadcast:generation-2",
        "abort:{'cell': 'generation-2'}",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_error", [RuntimeError("stop failed"), TimeoutError("stop timed out")])
async def test_an_ambiguous_participant_replacement_never_starts_a_second_attempt(abort_error: Exception):
    """An unconfirmed participant stop makes another window unsafe."""
    from miles.ray import placement_group
    from miles.utils.retry_utils import NonRetryableError

    order: list[str] = []
    inference_controller = _RetryWindowController(order, abort_errors=[abort_error])

    with pytest.raises(NonRetryableError, match="did not close cleanly"):
        await asyncio.wait_for(
            placement_group.update_weights(
                _orchestration_args(),
                MagicMock(update_weights=AsyncMock(side_effect=RuntimeError("broadcast failed"))),
                MagicMock(set_weight_version=AsyncMock()),
                inference_controller,
            ),
            timeout=1,
        )

    assert order == [
        "start:generation-1",
        "abort:{'cell': 'generation-1'}",
        *(["wait_idle:inference"] if isinstance(abort_error, TimeoutError) else []),
    ]


@pytest.mark.asyncio
async def test_a_terminal_trainer_failure_is_detected_before_opening_a_second_window():
    """A replacement readiness failure is terminal locally and cannot consume another engine snapshot."""
    from miles.ray import placement_group
    from miles.utils.retry_utils import NonRetryableError

    order: list[str] = []
    inference_controller = _RetryWindowController(order)
    actor_model = MagicMock(
        update_weights=AsyncMock(side_effect=RuntimeError("dead trainer")),
        wait_until_update_weights_ready=AsyncMock(side_effect=RuntimeError("remote terminal failure")),
    )

    with pytest.raises(NonRetryableError, match="did not close cleanly"):
        await asyncio.wait_for(
            placement_group.update_weights(
                _orchestration_args(),
                actor_model,
                MagicMock(set_weight_version=AsyncMock()),
                inference_controller,
            ),
            timeout=1,
        )

    assert order == ["start:generation-1", "abort:{'cell': 'generation-1'}"]
    actor_model.wait_until_update_weights_ready.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_drained_failure_after_cancellation_never_starts_a_second_window():
    """A failed drained child preserves its failure as non-retryable after the caller cancels."""
    from miles.ray import placement_group
    from miles.utils.retry_utils import NonRetryableError

    order: list[str] = []
    inference_controller = _RetryWindowController(order)
    broadcast_started = asyncio.Event()
    broadcast_release = asyncio.Event()

    async def _broadcast(*, info: UpdatableEngines, rollout_id: int | None = None) -> int:
        broadcast_started.set()
        await broadcast_release.wait()
        raise RuntimeError("drained broadcast failed")

    task = asyncio.create_task(
        placement_group.update_weights(
            _orchestration_args(),
            MagicMock(update_weights=AsyncMock(side_effect=_broadcast)),
            MagicMock(set_weight_version=AsyncMock()),
            inference_controller,
        )
    )
    await asyncio.wait_for(broadcast_started.wait(), timeout=1)

    task.cancel()
    await asyncio.sleep(0)
    broadcast_release.set()

    with pytest.raises(NonRetryableError, match="failed after caller cancellation") as error:
        await asyncio.wait_for(task, timeout=1)

    assert isinstance(error.value.__cause__, placement_group._RetryableWeightUpdateError)
    assert order == ["start:generation-1", "abort:{'cell': 'generation-1'}"]


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
    async def _log(args: Namespace, *, response=None, initialized: bool = True) -> tuple[MagicMock, MagicMock]:
        from miles.ray.placement_group import _maybe_log_inference_engine_weight_checksums

        inference_controller = MagicMock()
        inference_controller.check_weights = AsyncMock(return_value=response) if response is not None else MagicMock()
        event_logger = MagicMock()
        with patch("miles.ray.placement_group.is_event_logger_initialized", return_value=initialized), patch(
            "miles.ray.placement_group.get_event_logger", return_value=event_logger
        ):
            await _maybe_log_inference_engine_weight_checksums(
                args, inference_controller=inference_controller, rollout_id=0, trainer_model_id=None
            )
        return inference_controller, event_logger

    async def test_no_event_logger_does_not_call_check_weights(self):
        """Without an initialized event logger, no check_weights request is issued."""
        inference_controller, _ = await self._log(_orchestration_args(), initialized=False)

        inference_controller.check_weights.assert_not_called()

    async def test_flag_off_skips_collection(self):
        """Without --save-inference-engine-weight-checksum, no check_weights request is issued."""
        inference_controller, _ = await self._log(_orchestration_args(save_inference_engine_weight_checksum=False))

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
            rollout_id=0, engine_checksums=[{"rank0/w": "e0"}, {"rank0/w": "e1"}]
        )
