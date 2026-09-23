import asyncio
import logging

import pytest

from miles.utils import context_lock
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)


class _Journal:
    """Records who ran when, so a test can tell interleaving apart from serialization."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.max_open_sections = 0
        self._open_sections = 0

    def record(self, event: str) -> None:
        self.events.append(event)

    def enter_section(self, name: str) -> None:
        self._open_sections += 1
        self.max_open_sections = max(self.max_open_sections, self._open_sections)
        self.record(f"{name}-enter")

    def leave_section(self, name: str) -> None:
        self._open_sections -= 1
        self.record(f"{name}-leave")

    def events_between(self, start: str, end: str) -> list[str]:
        return self.events[self.events.index(start) + 1 : self.events.index(end)]


@enforce_lock_discipline
class _Server:
    """Shaped like RolloutServer: fully guarded, and handed its owner's lock."""

    @lock_exempt
    def __init__(self, journal: _Journal, lock: ContextLock, cell_ids: list[str]) -> None:
        self.journal = journal
        self.context_lock = lock
        self.cells = {cell_id: f"engine-{cell_id}" for cell_id in cell_ids}

    @requires_lock
    async def offload(self) -> None:
        await self._touch_every_cell("offload")

    @requires_lock
    async def onload(self) -> None:
        await self._touch_every_cell("onload")

    @requires_lock
    async def mark_weights_ready(self) -> None:
        await self._touch_every_cell("mark-weights-ready")

    @requires_lock
    async def remove_cell(self, cell_id: str) -> None:
        self.journal.record(f"remove:{cell_id}")
        await asyncio.sleep(0)
        self.cells.pop(cell_id, None)

    @property
    @requires_lock
    def engines(self) -> list[str]:
        return list(self.cells.values())

    @requires_lock
    async def _touch_every_cell(self, action: str) -> None:
        for cell_id in list(self.cells):
            self.journal.record(f"{action}:{cell_id}")
            await asyncio.sleep(0)


@enforce_lock_discipline
class _Controller:
    """Shaped like InferenceController: owns the lock and hands it to its server."""

    @lock_exempt
    def __init__(self, journal: _Journal, cell_ids: list[str] | None = None) -> None:
        self.journal = journal
        self.context_lock = ContextLock("InferenceController")
        self.server = _Server(journal, self.context_lock, cell_ids or ["cell-0", "cell-1"])

    @with_lock
    async def prepare_rollout(self) -> None:
        await self.server.onload()

    @with_lock
    async def offload(self) -> None:
        await self.server.offload()

    @with_lock
    async def onload_weights(self) -> None:
        await self._onload()

    @with_lock
    async def onload_kv(self) -> None:
        await self._onload()

    @with_lock
    async def reconcile(self, cell_id: str) -> None:
        await self.server.remove_cell(cell_id)

    @acquires_lock
    async def start_update_weights(self) -> list[str]:
        return self.server.engines

    @releases_lock
    async def end_update_weights(self, engines: list[str]) -> None:
        await self.server.mark_weights_ready()

    @requires_lock
    async def _onload(self) -> None:
        await self.server.onload()


async def _update_weights_flow(controller: _Controller, label: str) -> None:
    """Shaped like the trainer: snapshot the engines, broadcast, then mark them ready."""
    engines = await controller.start_update_weights()
    controller.journal.enter_section(label)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    controller.journal.leave_section(label)
    await controller.end_update_weights(engines)


class TestCallersRacingByAccident:
    @pytest.mark.asyncio
    async def test_two_overlapping_update_weights_flows_never_interleave(self):
        """A double-fired weight update (a retry, two trainers) must serialize, not interleave."""
        journal = _Journal()
        controller = _Controller(journal)

        await asyncio.gather(_update_weights_flow(controller, "first"), _update_weights_flow(controller, "second"))

        assert journal.max_open_sections == 1
        assert journal.events_between("first-enter", "first-leave") == []
        assert journal.events_between("second-enter", "second-leave") == []

    @pytest.mark.asyncio
    async def test_a_reconcile_cannot_land_inside_an_update_weights_window(self):
        """A cell vanishing mid-update must not be torn down under the trainer's snapshot."""
        journal = _Journal()
        controller = _Controller(journal)

        await asyncio.gather(_update_weights_flow(controller, "update"), controller.reconcile("cell-0"))

        assert journal.events_between("update-enter", "update-leave") == []
        assert "remove:cell-0" in journal.events

    @pytest.mark.asyncio
    async def test_an_offload_cannot_land_inside_an_update_weights_window(self):
        """Offloading engines whose weights are being written would corrupt the update."""
        journal = _Journal()
        controller = _Controller(journal)

        await asyncio.gather(_update_weights_flow(controller, "update"), controller.offload())

        assert not [event for event in journal.events_between("update-enter", "update-leave") if "offload" in event]

    @pytest.mark.asyncio
    async def test_offload_and_onload_fired_together_do_not_interleave_per_cell(self):
        """A forgotten await upstream can fire both at once; each engine still sees them in order."""
        journal = _Journal()
        controller = _Controller(journal, cell_ids=["cell-0", "cell-1"])

        await asyncio.gather(controller.offload(), controller.prepare_rollout())

        assert journal.events in (
            ["offload:cell-0", "offload:cell-1", "onload:cell-0", "onload:cell-1"],
            ["onload:cell-0", "onload:cell-1", "offload:cell-0", "offload:cell-1"],
        )

    @pytest.mark.asyncio
    async def test_a_whole_burst_of_lifecycle_calls_is_fully_serialized(self):
        """Whatever mix of calls arrives at once, exactly one critical section runs at a time."""
        journal = _Journal()
        controller = _Controller(journal)

        await asyncio.gather(
            _update_weights_flow(controller, "update-a"),
            _update_weights_flow(controller, "update-b"),
            controller.offload(),
            controller.onload_weights(),
            controller.prepare_rollout(),
            controller.reconcile("cell-1"),
        )

        assert journal.max_open_sections == 1
        assert not controller.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_engine_snapshot_cannot_change_while_the_window_is_open(self):
        """The whole point of the window: the engine list handed to the trainer stays valid."""
        journal = _Journal()
        controller = _Controller(journal)

        engines = await controller.start_update_weights()
        reconcile = asyncio.create_task(controller.reconcile("cell-0"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert set(controller.server.cells) == {"cell-0", "cell-1"}
        assert engines == ["engine-cell-0", "engine-cell-1"]

        await controller.end_update_weights(engines)
        await reconcile
        assert set(controller.server.cells) == {"cell-1"}


class TestMistakesAroundTheUpdateWeightsWindow:
    @pytest.mark.asyncio
    async def test_ending_the_window_twice_is_rejected(self):
        """A retry loop calling end twice must not release a window it no longer owns."""
        controller = _Controller(_Journal())
        engines = await controller.start_update_weights()
        await controller.end_update_weights(engines)

        with pytest.raises(AssertionError, match="was not detached"):
            await controller.end_update_weights(engines)

    @pytest.mark.asyncio
    async def test_ending_a_window_that_was_never_started_is_rejected(self):
        """An early return that skips start must not let end unlock the controller."""
        controller = _Controller(_Journal())

        with pytest.raises(AssertionError, match="was not detached"):
            await controller.end_update_weights([])

    @pytest.mark.asyncio
    async def test_a_forgotten_end_blocks_the_next_caller_instead_of_running_unguarded(self):
        """Losing the end call stalls the next rollout rather than letting it run unprotected."""
        controller = _Controller(_Journal())
        await controller.start_update_weights()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(controller.offload(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_a_stalled_window_is_reported_in_the_log(self, monkeypatch, caplog):
        """A trainer that died mid-update leaves the lock held, so the waiter must say so."""
        monkeypatch.setattr(context_lock, "WAIT_LOG_INTERVAL_SECONDS", 0.01)
        controller = _Controller(_Journal())
        await controller.start_update_weights()

        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            blocked = asyncio.create_task(controller.offload())
            await asyncio.sleep(0.05)
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked

        assert any("Still waiting for lock 'InferenceController'" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_start_that_raises_does_not_wedge_the_controller(self):
        """When taking the snapshot fails, the next rollout must still be able to proceed."""

        @enforce_lock_discipline
        class _FailingController(_Controller):
            @acquires_lock
            async def start_update_weights(self) -> list[str]:
                raise RuntimeError("engines never became ready")

        controller = _FailingController(_Journal())
        with pytest.raises(RuntimeError, match="engines never became ready"):
            await controller.start_update_weights()

        assert not controller.context_lock.locked
        await controller.offload()


class TestBypassingTheOwner:
    @pytest.mark.asyncio
    async def test_driving_the_server_directly_is_rejected(self):
        """New code reaching into controller.server instead of going through the controller."""
        controller = _Controller(_Journal())

        with pytest.raises(AssertionError, match="must be called with"):
            await controller.server.offload()

    def test_reading_the_engine_snapshot_directly_is_rejected(self):
        """A snapshot read outside the lock may already be stale by the time it is used."""
        controller = _Controller(_Journal())

        with pytest.raises(AssertionError, match="must be called with"):
            _ = controller.server.engines

    @pytest.mark.asyncio
    async def test_holding_some_other_lock_does_not_authorize_the_server(self):
        """Grabbing a nearby lock instead of the owning controller's is still a bypass."""
        controller = _Controller(_Journal())

        async with ContextLock("SomeOtherController"):
            with pytest.raises(AssertionError, match="must be called with"):
                await controller.server.offload()

    @pytest.mark.asyncio
    async def test_a_server_built_with_its_own_fresh_lock_is_rejected(self):
        """The classic wiring mistake: constructing a lock instead of accepting the owner's."""
        journal = _Journal()
        controller = _Controller(journal)
        controller.server = _Server(journal, ContextLock("InferenceController"), ["cell-0"])

        with pytest.raises(AssertionError, match="must be called with"):
            await controller.offload()


class TestMistakesWhenExtendingAGuardedClass:
    def test_adding_a_method_and_forgetting_the_decorator_is_rejected_at_class_creation(self):
        """The common mistake: a new method whose author never thought about the lock."""
        with pytest.raises(AssertionError, match="drain_engines must be decorated"):

            @enforce_lock_discipline
            class _Extended(_Controller):
                async def drain_engines(self) -> None:
                    pass

    @pytest.mark.asyncio
    async def test_decorating_the_method_but_forgetting_the_class_is_rejected_at_call_time(self):
        """The subtler mistake: the method states its discipline but nothing ever checks the class."""

        class _Extended(_Controller):
            @with_lock
            async def drain_engines(self) -> None:
                pass

        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await _Extended(_Journal()).drain_engines()

    @pytest.mark.asyncio
    async def test_a_locked_method_calling_a_sibling_locked_method_is_rejected(self):
        """The reentrancy mistake: delegating to a public sibling instead of a private helper."""

        @enforce_lock_discipline
        class _Extended(_Controller):
            @with_lock
            async def onload_everything(self) -> None:
                await self.onload_weights()

        with pytest.raises(AssertionError, match="already held"):
            await _Extended(_Journal()).onload_everything()

    @pytest.mark.asyncio
    async def test_delegating_to_a_private_helper_instead_is_the_shape_that_works(self):
        """Regression for that fix: sibling entry points share one lock-requiring private helper."""
        controller = _Controller(_Journal())

        await controller.onload_weights()
        await controller.onload_kv()

        assert not controller.context_lock.locked

    @pytest.mark.asyncio
    async def test_a_guarded_helper_calling_another_guarded_helper_is_fine(self):
        """requires_lock nests freely; only re-acquiring the lock is the error."""
        controller = _Controller(_Journal())

        async with controller.context_lock:
            await controller._onload()
            await controller.server.offload()
