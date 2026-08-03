import asyncio
import dataclasses
import functools
import itertools
import logging
from types import SimpleNamespace

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


@enforce_lock_discipline
class _Guarded:
    @lock_exempt
    def __init__(self) -> None:
        self.context_lock = ContextLock("guarded")
        self.max_concurrent_calls = 0
        self._concurrent_calls = 0

    @with_lock
    async def locked_method(self, delay: float = 0) -> bool:
        self._concurrent_calls += 1
        self.max_concurrent_calls = max(self.max_concurrent_calls, self._concurrent_calls)
        await asyncio.sleep(delay)
        self._concurrent_calls -= 1
        return self.context_lock.held_in_current_context

    @with_lock
    async def locked_method_that_raises(self) -> None:
        raise RuntimeError("boom")

    @with_lock
    async def locked_method_returning(self, value: int) -> int:
        return value

    @with_lock
    async def locked_method_calling_private(self) -> bool:
        return self._private_method()

    @with_lock
    async def locked_method_fanning_out(self) -> list[bool]:
        return await asyncio.gather(self.async_private_method(), self.async_private_method())

    @requires_lock
    async def async_private_method(self) -> bool:
        return True

    @requires_lock
    def _private_method(self) -> bool:
        return True

    @property
    @requires_lock
    def guarded_value(self) -> int:
        return 42

    @acquires_lock
    async def start_window(self) -> int:
        return self.guarded_value

    @acquires_lock
    async def start_window_that_raises(self) -> None:
        raise RuntimeError("boom")

    @releases_lock
    async def end_window(self) -> int:
        return self.guarded_value


class _HolderTask:
    """Holds the lock in a separate task, so the test body stays a non-holding context."""

    def __init__(self, lock: ContextLock) -> None:
        self._lock = lock
        self._acquired = asyncio.Event()
        self._release_requested = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._hold())
        await self._acquired.wait()

    async def finish(self) -> None:
        self._release_requested.set()
        await self._task

    async def _hold(self) -> None:
        await self._lock.acquire()
        self._acquired.set()
        await self._release_requested.wait()
        self._lock.release()


class TestContextLock:
    @pytest.mark.asyncio
    async def test_acquire_and_release_toggle_locked_and_held_state(self):
        """acquire marks the lock locked and held by the current context; release clears both."""
        lock = ContextLock("test")
        await lock.acquire()
        assert lock.locked and lock.held_in_current_context
        lock.release()
        assert not lock.locked and not lock.held_in_current_context

    def test_the_lock_exposes_its_name(self):
        """Diagnostics identify which lock is involved."""
        assert ContextLock("InferenceController").name == "InferenceController"

    @pytest.mark.asyncio
    async def test_acquire_asserts_when_the_same_lock_is_already_held_in_context(self):
        """Reentrant acquisition is a bug, not a wait."""
        lock = ContextLock("test")
        await lock.acquire()
        with pytest.raises(AssertionError, match="already held"):
            await lock.acquire()

    @pytest.mark.asyncio
    async def test_acquire_asserts_when_another_lock_is_held_in_context(self):
        """Holding two context locks at once would make held-by ambiguous."""
        first_lock = ContextLock("first")
        second_lock = ContextLock("second")
        await first_lock.acquire()
        with pytest.raises(AssertionError, match="already held"):
            await second_lock.acquire()

    @pytest.mark.asyncio
    async def test_release_asserts_when_not_held_by_the_current_context(self):
        """A context that never acquired the lock cannot release it."""
        lock = ContextLock("test")
        with pytest.raises(AssertionError, match="must be held"):
            lock.release()

    @pytest.mark.asyncio
    async def test_release_asserts_when_the_lock_is_held_by_another_task(self):
        """Only the holding context may release, even though the lock is locked."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        with pytest.raises(AssertionError, match="must be held"):
            lock.release()
        await holder.finish()

    @pytest.mark.asyncio
    async def test_a_second_acquirer_waits_until_release(self):
        """A caller blocks on acquire until the holding task releases."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        waiter = asyncio.create_task(lock.acquire())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not waiter.done()

        await holder.finish()
        await waiter
        assert lock.locked

    @pytest.mark.asyncio
    async def test_held_state_is_not_visible_to_an_unrelated_context(self):
        """held_in_current_context tracks the holder, not the global locked flag."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        assert lock.locked
        assert not lock.held_in_current_context
        await holder.finish()

    @pytest.mark.asyncio
    async def test_holding_one_lock_does_not_look_like_holding_another(self):
        """Two locks are tracked as distinct objects, not as one global flag."""
        lock = ContextLock("test")
        other_lock = ContextLock("other")
        await lock.acquire()
        assert lock.held_in_current_context
        assert not other_lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_work_spawned_inside_the_critical_section_sees_the_lock_as_held(self):
        """Tasks fanned out from inside the section (e.g. asyncio.gather) are still inside it."""
        lock = ContextLock("test")
        async with lock:
            assert all(await asyncio.gather(_read_held(lock), _read_held(lock)))

    @pytest.mark.asyncio
    async def test_async_with_acquires_and_releases(self):
        """The lock is a context manager for plain lexical critical sections."""
        lock = ContextLock("test")
        async with lock:
            assert lock.locked and lock.held_in_current_context
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_async_with_releases_on_exception(self):
        """An exception inside the critical section must not leak a held lock."""
        lock = ContextLock("test")
        with pytest.raises(RuntimeError, match="boom"):
            async with lock:
                raise RuntimeError("boom")
        assert not lock.locked


class TestWithLock:
    @pytest.mark.asyncio
    async def test_the_lock_is_held_during_the_call_and_released_after(self):
        """with_lock wraps the method body in an acquire/release pair."""
        guarded = _Guarded()
        assert await guarded.locked_method() is True
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_lock_is_released_when_the_method_raises(self):
        """An exception inside the method must not leak a held lock."""
        guarded = _Guarded()
        with pytest.raises(RuntimeError, match="boom"):
            await guarded.locked_method_that_raises()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_serialized(self):
        """Two decorated calls never run their bodies at the same time."""
        guarded = _Guarded()
        await asyncio.gather(guarded.locked_method(delay=0.01), guarded.locked_method(delay=0.01))
        assert guarded.max_concurrent_calls == 1

    @pytest.mark.asyncio
    async def test_the_return_value_is_passed_through(self):
        """The wrapper is transparent for arguments and return values."""
        guarded = _Guarded()
        assert await guarded.locked_method_returning(42) == 42

    @pytest.mark.asyncio
    async def test_two_instances_sharing_a_lock_serialize_against_each_other(self):
        """Collaborators are expected to be handed the very same lock object."""
        first = _Guarded()
        second = _Guarded()
        second.context_lock = first.context_lock

        await asyncio.gather(first.locked_method(delay=0.01), second.locked_method(delay=0.01))
        assert first.max_concurrent_calls == 1
        assert second.max_concurrent_calls == 1

    def test_rejects_sync_functions_at_decoration_time(self):
        """with_lock cannot await inside a sync function, so it must refuse one."""
        with pytest.raises(AssertionError, match="must be async"):

            @with_lock
            def sync_method(self) -> None:
                pass

    @pytest.mark.asyncio
    async def test_a_nested_call_to_another_locked_method_asserts(self):
        """Lock reentrancy through decorated methods is surfaced as an assertion, not a deadlock."""

        @enforce_lock_discipline
        class _Nested:
            @lock_exempt
            def __init__(self) -> None:
                self.context_lock = ContextLock("nested")

            @with_lock
            async def outer(self) -> None:
                await self.inner()

            @with_lock
            async def inner(self) -> None:
                pass

        with pytest.raises(AssertionError, match="already held"):
            await _Nested().outer()

    @pytest.mark.asyncio
    async def test_a_missing_lock_attribute_is_reported(self):
        """A class that never got handed its lock fails loudly at call time."""

        @enforce_lock_discipline
        class _NoLock:
            @with_lock
            async def method(self) -> None:
                pass

        with pytest.raises(AttributeError, match="context_lock"):
            await _NoLock().method()

    @pytest.mark.asyncio
    async def test_a_lock_attribute_of_the_wrong_type_is_reported(self):
        """Handing over a bare asyncio.Lock loses the held-by tracking, so it is refused."""

        @enforce_lock_discipline
        class _WrongLock:
            @lock_exempt
            def __init__(self) -> None:
                self.context_lock = asyncio.Lock()

            @with_lock
            async def method(self) -> None:
                pass

        with pytest.raises(AssertionError, match="must be a ContextLock"):
            await _WrongLock().method()


async def _read_held(lock: ContextLock) -> bool:
    return lock.held_in_current_context


async def _reattach_and_release(lock: ContextLock) -> None:
    lock.reattach()
    lock.release()


async def _acquire_release_and_report(lock: ContextLock, taken: list) -> None:
    await lock.acquire()
    taken.append(lock.held_in_current_context)
    lock.release()


async def _acquire_and_report_held(lock: ContextLock) -> bool:
    await lock.acquire()
    return lock.held_in_current_context


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _reminder_messages(caplog) -> list[str]:
    return [record.message for record in caplog.records if record.message.startswith("Still waiting for lock")]


class TestWaitReminder:
    @pytest.fixture
    def fast_reminder(self, monkeypatch):
        monkeypatch.setattr(context_lock, "WAIT_LOG_INTERVAL_SECONDS", 0.01)

    @pytest.mark.asyncio
    async def test_a_blocked_acquirer_reminds_repeatedly(self, fast_reminder, caplog):
        """A caller stuck on the lock keeps reporting that it is still waiting."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await asyncio.sleep(0.08)
        await _cancel(waiter)
        await holder.finish()

        assert len(_reminder_messages(caplog)) >= 2

    @pytest.mark.asyncio
    async def test_the_reminder_names_the_lock_and_the_elapsed_time(self, fast_reminder, caplog):
        """The operator needs to know which lock is stuck and for how long."""
        lock = ContextLock("stuck-lock")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await asyncio.sleep(0.05)
        await _cancel(waiter)
        await holder.finish()

        messages = _reminder_messages(caplog)
        assert messages
        assert all(message.startswith("Still waiting for lock 'stuck-lock' after ") for message in messages)
        assert all(message.endswith("s") for message in messages)

    @pytest.mark.asyncio
    async def test_the_reminder_counts_up_while_the_wait_drags_on(self, fast_reminder, caplog, monkeypatch):
        """A reminder that always says the same thing cannot tell a short stall from a stuck one."""
        ticks = itertools.count(start=0, step=6)
        monkeypatch.setattr(context_lock, "time", SimpleNamespace(monotonic=lambda: float(next(ticks))))
        lock = ContextLock("stuck-lock")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await asyncio.sleep(0.05)
        await _cancel(waiter)
        await holder.finish()

        assert _reminder_messages(caplog)[:2] == [
            "Still waiting for lock 'stuck-lock' after 6s",
            "Still waiting for lock 'stuck-lock' after 12s",
        ]

    @pytest.mark.asyncio
    async def test_an_uncontended_acquire_reminds_nothing(self, fast_reminder, caplog):
        """The reminder must only fire while actually blocked."""
        lock = ContextLock("test")
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await lock.acquire()
            await asyncio.sleep(0.05)
            lock.release()

        assert _reminder_messages(caplog) == []

    @pytest.mark.asyncio
    async def test_the_reminder_stops_once_the_lock_is_acquired(self, fast_reminder, caplog):
        """No reminder may keep firing after the caller got in."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0.03)
        await holder.finish()
        await waiter

        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await asyncio.sleep(0.05)

        assert _reminder_messages(caplog) == []

    @pytest.mark.asyncio
    async def test_a_waiter_that_eventually_wins_acquires_normally(self, fast_reminder):
        """Reminding must not interfere with the acquisition itself."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(_acquire_and_report_held(lock))
        await asyncio.sleep(0.05)
        await holder.finish()

        assert await waiter is True
        assert lock.locked

    @pytest.mark.asyncio
    async def test_a_cancelled_waiter_leaves_the_lock_untouched(self, fast_reminder):
        """Giving up on the lock must neither mark it held nor steal it."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0.03)
        await _cancel(waiter)
        await holder.finish()

        assert not lock.locked

    @pytest.mark.asyncio
    async def test_a_cancelled_waiter_stops_reminding(self, fast_reminder, caplog):
        """A leaked reminder would keep claiming a caller is waiting long after it gave up."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()

        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0.03)
        await _cancel(waiter)
        with caplog.at_level(logging.INFO, logger="miles.utils.context_lock"):
            await asyncio.sleep(0.05)
        await holder.finish()

        assert _reminder_messages(caplog) == []


class TestDetachAndReattach:
    @pytest.mark.asyncio
    async def test_detach_keeps_the_lock_locked_but_not_held(self):
        """A detached lock stays locked so other contexts keep blocking."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        assert lock.locked and not lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_detach_asserts_when_not_held_by_the_current_context(self):
        """Only the holding context may hand the lock over."""
        lock = ContextLock("test")
        with pytest.raises(AssertionError, match="must be held"):
            lock.detach()

    @pytest.mark.asyncio
    async def test_a_detached_lock_still_blocks_other_acquirers(self):
        """Detaching hands the lock across calls; it does not open it up."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        waiter = asyncio.create_task(lock.acquire())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not waiter.done()

        lock.reattach()
        lock.release()
        await waiter

    @pytest.mark.asyncio
    async def test_a_detached_lock_can_be_reattached_and_released_by_another_task(self):
        """The start/end pair of a cross-call window may run in different tasks."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        await asyncio.create_task(_reattach_and_release(lock))
        assert not lock.locked

    @pytest.mark.asyncio
    async def test_reattach_asserts_when_the_lock_was_not_detached(self):
        """reattach must not steal a lock that is held normally by someone else."""
        lock = ContextLock("test")
        holder = _HolderTask(lock)
        await holder.start()
        with pytest.raises(AssertionError, match="was not detached"):
            lock.reattach()
        await holder.finish()

    @pytest.mark.asyncio
    async def test_reattach_asserts_when_a_lock_is_already_held_in_context(self):
        """A context cannot adopt a detached lock while holding another lock."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        other_lock = ContextLock("other")
        await other_lock.acquire()
        with pytest.raises(AssertionError, match="already held"):
            lock.reattach()

    @pytest.mark.asyncio
    async def test_reattach_restores_the_held_state(self):
        """After reattach the context may again call lock-requiring helpers."""
        lock = ContextLock("test")
        await lock.acquire()
        lock.detach()
        lock.reattach()
        assert lock.held_in_current_context
        lock.release()
        assert not lock.locked


class TestWithReleased:
    @pytest.mark.asyncio
    async def test_the_lock_is_open_inside_the_block(self):
        """A long wait must not keep everyone else out while it polls."""
        lock = ContextLock("test")
        await lock.acquire()

        async with lock.with_released():
            assert not lock.locked and not lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_the_lock_is_held_again_after_the_block(self):
        """The caller was inside a locked region, so it must get its lock back."""
        lock = ContextLock("test")
        await lock.acquire()

        async with lock.with_released():
            pass

        assert lock.locked and lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_another_task_can_take_the_lock_inside_the_block(self):
        """This is the whole point: the waiter lets the work it waits for proceed."""
        lock = ContextLock("test")
        await lock.acquire()
        taken = []

        async with lock.with_released():
            await asyncio.create_task(_acquire_release_and_report(lock, taken))

        assert taken == [True]

    @pytest.mark.asyncio
    async def test_the_block_waits_for_a_concurrent_holder_before_returning(self):
        """Re-acquiring must queue behind whoever took the lock, not steal it."""
        lock = ContextLock("test")
        await lock.acquire()
        holder = _HolderTask(lock)

        async with lock.with_released():
            await holder.start()
            assert lock.locked and not lock.held_in_current_context
            await holder.finish()

        assert lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_a_raising_block_still_gets_the_lock_back(self):
        """The caller's own lock discipline continues after the failure is handled."""
        lock = ContextLock("test")
        await lock.acquire()

        with pytest.raises(RuntimeError, match="boom"):
            async with lock.with_released():
                raise RuntimeError("boom")

        assert lock.held_in_current_context

    @pytest.mark.asyncio
    async def test_a_caller_that_does_not_hold_the_lock_is_rejected(self):
        """Releasing a lock one does not hold would open someone else's critical section."""
        lock = ContextLock("test")

        with pytest.raises(AssertionError, match="must be held"):
            async with lock.with_released():
                pass

    @pytest.mark.asyncio
    async def test_lock_requiring_helpers_are_rejected_inside_the_block(self):
        """Inside the block the invariants the lock guards no longer hold."""
        guarded = _Guarded()
        await guarded.context_lock.acquire()

        async with guarded.context_lock.with_released():
            with pytest.raises(AssertionError, match="must be called with"):
                guarded._private_method()


class TestAcquiresAndReleasesLock:
    @pytest.mark.asyncio
    async def test_the_lock_stays_locked_between_the_start_and_end_calls(self):
        """acquires_lock opens a cross-call window that releases_lock closes."""
        guarded = _Guarded()
        await guarded.start_window()
        assert guarded.context_lock.locked

        await guarded.end_window()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_lock_is_held_inside_both_the_start_and_end_bodies(self):
        """Both window methods may call lock-requiring helpers in their bodies."""
        guarded = _Guarded()
        assert await guarded.start_window() == 42
        assert await guarded.end_window() == 42

    @pytest.mark.asyncio
    async def test_the_lock_is_not_held_by_the_caller_between_the_two_calls(self):
        """The window belongs to the lock, not to whoever happens to call end."""
        guarded = _Guarded()
        await guarded.start_window()
        assert not guarded.context_lock.held_in_current_context

        await guarded.end_window()

    @pytest.mark.asyncio
    async def test_lock_requiring_helpers_are_rejected_between_the_two_calls(self):
        """An open window is not an invitation to touch guarded state from outside."""
        guarded = _Guarded()
        await guarded.start_window()
        with pytest.raises(AssertionError, match="must be called with"):
            guarded._private_method()

        await guarded.end_window()

    @pytest.mark.asyncio
    async def test_the_lock_is_released_when_the_start_call_raises(self):
        """A failed start must not leave the lock stuck forever."""
        guarded = _Guarded()
        with pytest.raises(RuntimeError, match="boom"):
            await guarded.start_window_that_raises()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_a_failed_start_leaves_the_lock_acquirable_again(self):
        """After the failure another caller may open a window normally."""
        guarded = _Guarded()
        with pytest.raises(RuntimeError, match="boom"):
            await guarded.start_window_that_raises()

        await guarded.start_window()
        await guarded.end_window()
        assert not guarded.context_lock.locked

    @pytest.mark.asyncio
    async def test_the_end_call_asserts_without_a_matching_start(self):
        """releases_lock without an open window is a bug."""
        guarded = _Guarded()
        with pytest.raises(AssertionError, match="was not detached"):
            await guarded.end_window()

    @pytest.mark.asyncio
    async def test_concurrent_locked_calls_block_during_the_window(self):
        """with_lock methods wait out an open window instead of interleaving with it."""
        guarded = _Guarded()
        await guarded.start_window()
        blocked_call = asyncio.create_task(guarded.locked_method())
        for _ in range(5):
            await asyncio.sleep(0)
        assert not blocked_call.done()

        await guarded.end_window()
        assert await blocked_call is True

    @pytest.mark.asyncio
    async def test_the_window_can_be_opened_and_closed_from_different_tasks(self):
        """start and end are separate scheduler invocations in the real caller."""
        guarded = _Guarded()
        await asyncio.create_task(guarded.start_window())
        assert guarded.context_lock.locked

        await asyncio.create_task(guarded.end_window())
        assert not guarded.context_lock.locked

    def test_rejects_sync_functions_at_decoration_time(self):
        """Both window decorators need to await the lock, so sync functions are refused."""
        with pytest.raises(AssertionError, match="must be async"):

            @acquires_lock
            def sync_start(self) -> None:
                pass

        with pytest.raises(AssertionError, match="must be async"):

            @releases_lock
            def sync_end(self) -> None:
                pass


class TestRequiresLock:
    @pytest.mark.asyncio
    async def test_passes_when_called_from_a_lock_holding_method(self):
        """Private helpers run fine inside a with_lock caller."""
        guarded = _Guarded()
        assert await guarded.locked_method_calling_private() is True

    @pytest.mark.asyncio
    async def test_passes_inside_an_explicit_lock_context(self):
        """Holding the lock via async with also satisfies the requirement."""
        guarded = _Guarded()
        async with guarded.context_lock:
            assert await guarded.async_private_method() is True
            assert guarded._private_method() is True
            assert guarded.guarded_value == 42

    @pytest.mark.asyncio
    async def test_passes_in_tasks_fanned_out_from_inside_the_lock(self):
        """asyncio.gather from a locked method must not trip the check in its children."""
        guarded = _Guarded()
        assert await guarded.locked_method_fanning_out() == [True, True]

    @pytest.mark.asyncio
    async def test_raises_when_no_lock_is_held(self):
        """Calling a lock-requiring method outside the lock is rejected."""
        guarded = _Guarded()
        with pytest.raises(AssertionError, match="must be called with the 'guarded' context lock held"):
            guarded._private_method()

    @pytest.mark.asyncio
    async def test_raises_for_async_methods_before_running_the_body(self):
        """The decorator asserts before awaiting async bodies too."""
        guarded = _Guarded()
        with pytest.raises(AssertionError, match="must be called with"):
            await guarded.async_private_method()

    @pytest.mark.asyncio
    async def test_raises_for_property_access_outside_the_lock(self):
        """Guarded snapshots must not be readable without the lock."""
        guarded = _Guarded()
        with pytest.raises(AssertionError, match="must be called with"):
            _ = guarded.guarded_value

    @pytest.mark.asyncio
    async def test_raises_when_a_different_lock_is_held(self):
        """Holding some unrelated lock does not authorize touching this object."""
        guarded = _Guarded()
        async with ContextLock("unrelated"):
            with pytest.raises(AssertionError, match="must be called with the 'guarded' context lock held"):
                guarded._private_method()

    @pytest.mark.asyncio
    async def test_passes_when_a_collaborator_shares_the_very_same_lock(self):
        """Collaborators guarded by one controller are handed that controller's lock object."""
        controller = _Guarded()
        collaborator = _Guarded()
        collaborator.context_lock = controller.context_lock

        async with controller.context_lock:
            assert collaborator._private_method() is True

    @pytest.mark.asyncio
    async def test_raises_when_a_collaborator_holds_a_look_alike_lock(self):
        """A separate lock object with the same name is still the wrong lock."""
        controller = _Guarded()
        collaborator = _Guarded()

        async with controller.context_lock:
            with pytest.raises(AssertionError, match="must be called with"):
                collaborator._private_method()

    @pytest.mark.asyncio
    async def test_raises_when_the_lock_is_held_by_another_task(self):
        """held-by matters: someone else holding the lock does not authorize this context."""
        guarded = _Guarded()
        holder = _HolderTask(guarded.context_lock)
        await holder.start()
        with pytest.raises(AssertionError, match="must be called with"):
            guarded._private_method()
        await holder.finish()

    @pytest.mark.asyncio
    async def test_the_requirement_lapses_again_after_the_holder_returns(self):
        """The requirement is scoped to the critical section, not sticky afterwards."""
        guarded = _Guarded()
        assert await guarded.locked_method_calling_private() is True
        with pytest.raises(AssertionError, match="must be called with"):
            guarded._private_method()

    @pytest.mark.asyncio
    async def test_reports_a_missing_lock_attribute(self):
        """A collaborator that never got handed the lock fails loudly."""

        @enforce_lock_discipline
        class _NoLock:
            @requires_lock
            def method(self) -> None:
                pass

        with pytest.raises(AttributeError, match="context_lock"):
            _NoLock().method()


class TestEnforceLockDiscipline:
    def test_rejects_an_undecorated_async_method(self):
        """Every method must opt into a lock discipline explicitly."""
        with pytest.raises(AssertionError, match="_Bad.method must be decorated"):

            @enforce_lock_discipline
            class _Bad:
                async def method(self) -> None:
                    pass

    def test_rejects_an_undecorated_sync_method(self):
        """Sync methods are checked, not just coroutines."""
        with pytest.raises(AssertionError, match="_BadSync.method must be decorated"):

            @enforce_lock_discipline
            class _BadSync:
                def method(self) -> None:
                    pass

    def test_rejects_undecorated_private_methods(self):
        """Private methods must be disciplined too, not just the public surface."""
        with pytest.raises(AssertionError, match="_BadPrivate._method must be decorated"):

            @enforce_lock_discipline
            class _BadPrivate:
                def _method(self) -> None:
                    pass

    def test_rejects_an_undecorated_dunder_method(self):
        """Dunders are methods too: a hand-written __init__ must state its discipline."""
        with pytest.raises(AssertionError, match=r"_BadInit.__init__ must be decorated"):

            @enforce_lock_discipline
            class _BadInit:
                def __init__(self) -> None:
                    pass

    def test_rejects_an_undecorated_dunder_other_than_init(self):
        """Any dunder can touch guarded state, so none of them are waved through."""
        with pytest.raises(AssertionError, match=r"_BadRepr.__repr__ must be decorated"):

            @enforce_lock_discipline
            class _BadRepr:
                def __repr__(self) -> str:
                    return "bad"

    def test_rejects_an_undecorated_property_getter(self):
        """Property getters are methods too and must be disciplined."""
        with pytest.raises(AssertionError, match="_BadProperty.value must be decorated"):

            @enforce_lock_discipline
            class _BadProperty:
                @property
                def value(self) -> int:
                    return 0

    def test_rejects_an_undecorated_property_setter(self):
        """A disciplined getter does not excuse an undisciplined setter."""
        with pytest.raises(AssertionError, match="_BadSetter.value must be decorated"):

            @enforce_lock_discipline
            class _BadSetter:
                @property
                @lock_exempt
                def value(self) -> int:
                    return 0

                @value.setter
                def value(self, new_value: int) -> None:
                    pass

    def test_rejects_an_undecorated_staticmethod(self):
        """Static methods are unwrapped and checked like the rest."""
        with pytest.raises(AssertionError, match="_BadStatic.method must be decorated"):

            @enforce_lock_discipline
            class _BadStatic:
                @staticmethod
                def method() -> None:
                    pass

    def test_rejects_an_undecorated_classmethod(self):
        """Class methods are unwrapped and checked like the rest."""
        with pytest.raises(AssertionError, match="_BadClassmethod.method must be decorated"):

            @enforce_lock_discipline
            class _BadClassmethod:
                @classmethod
                def method(cls) -> None:
                    pass

    def test_accepts_a_fully_disciplined_class(self):
        """A class whose members are all decorated or exempt passes the check."""

        @enforce_lock_discipline
        class _Good:
            @lock_exempt
            def __init__(self) -> None:
                self.context_lock = ContextLock("good")

            @staticmethod
            @lock_exempt
            def create() -> None:
                pass

            @classmethod
            @lock_exempt
            def build(cls) -> None:
                pass

            @with_lock
            async def method(self) -> None:
                pass

            @property
            @lock_exempt
            def value(self) -> int:
                return 0

        assert _Good().value == 0

    def test_accepts_a_dataclass_when_the_check_runs_before_field_generation(self):
        """Dataclass-generated dunders cannot be decorated, so the check must run inside the dataclass."""

        @dataclasses.dataclass
        @enforce_lock_discipline
        class _GoodDataclass:
            count: int = 0
            name: str = "default"

            @lock_exempt
            def method(self) -> int:
                return self.count

        assert _GoodDataclass(count=1).method() == 1

    def test_rejects_a_dataclass_whose_generated_dunders_get_checked(self):
        """Applying the check outside @dataclass sees generated methods it cannot possibly discipline."""
        with pytest.raises(AssertionError, match="_BadOrder.* must be decorated"):

            @enforce_lock_discipline
            @dataclasses.dataclass
            class _BadOrder:
                count: int = 0

    def test_accepts_plain_and_annotated_class_attributes(self):
        """Only real methods are checked; data attributes and their annotations are not."""

        @enforce_lock_discipline
        class _GoodAttributes:
            timeout_seconds: int = 30
            retries = 3

        assert _GoodAttributes.timeout_seconds == 30

    def test_accepts_a_class_whose_only_members_are_inherited(self):
        """A subclass adding nothing has nothing of its own to check."""

        @enforce_lock_discipline
        class _Derived(_Guarded):
            pass

        assert _Derived() is not None

    def test_the_guarded_test_double_passes_the_check(self):
        """The _Guarded helper used across this file is itself fully disciplined."""
        assert enforce_lock_discipline(_Guarded) is _Guarded

    @pytest.mark.asyncio
    async def test_a_with_lock_method_on_an_undecorated_class_is_rejected(self):
        """Decorating methods but forgetting the class decorator is exactly the missed check to catch."""

        class _Forgotten:
            def __init__(self) -> None:
                self.context_lock = ContextLock("forgotten")

            @with_lock
            async def method(self) -> None:
                pass

        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await _Forgotten().method()

    @pytest.mark.asyncio
    async def test_a_requires_lock_method_on_an_undecorated_class_is_rejected_inside_the_lock(self):
        """The missing class decorator is reported even where the lock check itself would pass."""

        class _Forgotten:
            def __init__(self) -> None:
                self.context_lock = ContextLock("forgotten")

            @requires_lock
            def method(self) -> None:
                pass

        forgotten = _Forgotten()
        async with forgotten.context_lock:
            with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
                forgotten.method()

    @pytest.mark.asyncio
    async def test_window_decorators_on_an_undecorated_class_are_rejected(self):
        """Cross-call window decorators demand the class-level check as well."""

        class _Forgotten:
            def __init__(self) -> None:
                self.context_lock = ContextLock("forgotten")

            @acquires_lock
            async def start(self) -> None:
                pass

            @releases_lock
            async def end(self) -> None:
                pass

        forgotten = _Forgotten()
        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await forgotten.start()
        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await forgotten.end()

    @pytest.mark.asyncio
    async def test_an_undecorated_subclass_may_still_call_inherited_methods(self):
        """The check applies to the class that declared the method, so inheriting is fine."""

        class _Derived(_Guarded):
            pass

        assert await _Derived().locked_method() is True

    @pytest.mark.asyncio
    async def test_a_subclass_adding_its_own_decorated_method_needs_its_own_check(self):
        """An inherited flag must not excuse methods newly decorated on the subclass."""

        class _Derived(_Guarded):
            @with_lock
            async def extra_method(self) -> None:
                pass

        derived = _Derived()
        assert await derived.locked_method() is True
        with pytest.raises(AssertionError, match="_Derived.extra_method uses a context-lock decorator"):
            await derived.extra_method()

    @pytest.mark.asyncio
    async def test_a_guarded_classmethod_on_an_undecorated_class_is_rejected(self):
        """Looking the owner up from the first argument silently let this shape through before."""

        class _Forgotten:
            context_lock = ContextLock("forgotten")

            @classmethod
            @with_lock
            async def method(cls) -> None:
                pass

        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await _Forgotten.method()

    @pytest.mark.asyncio
    async def test_a_guarded_method_wrapped_by_another_decorator_is_still_checked(self):
        """functools.wraps copies the discipline marker, so the outer wrapper must not become invisible."""

        def _passthrough(fn):
            @functools.wraps(fn)
            async def outer(self, *args, **kwargs):
                return await fn(self, *args, **kwargs)

            return outer

        class _Forgotten:
            def __init__(self) -> None:
                self.context_lock = ContextLock("forgotten")

            @_passthrough
            @with_lock
            async def method(self) -> None:
                pass

        with pytest.raises(AssertionError, match="not decorated with @enforce_lock_discipline"):
            await _Forgotten().method()

    @pytest.mark.asyncio
    async def test_a_guarded_method_wrapped_by_another_decorator_runs_on_an_enforced_class(self):
        """An outer functools.wraps decorator must not hide the owner marker from the inner lock wrapper."""

        def _passthrough(fn):
            @functools.wraps(fn)
            async def outer(self, *args, **kwargs):
                return await fn(self, *args, **kwargs)

            return outer

        @enforce_lock_discipline
        class _Stacked:
            @lock_exempt
            def __init__(self) -> None:
                self.context_lock = ContextLock("stacked")

            @_passthrough
            @with_lock
            async def method(self) -> bool:
                return self.context_lock.held_in_current_context

        assert await _Stacked().method() is True

    @pytest.mark.asyncio
    async def test_a_guarded_method_shared_with_an_unenforced_class_is_still_rejected(self):
        """Stamping the inner wrapper must authorize the enforcing class only, not everyone reusing it."""

        @with_lock
        async def _shared(self) -> bool:
            return self.context_lock.held_in_current_context

        @enforce_lock_discipline
        class _Enforced:
            @lock_exempt
            def __init__(self) -> None:
                self.context_lock = ContextLock("enforced")

            method = _shared

        class _Forgotten:
            def __init__(self) -> None:
                self.context_lock = ContextLock("forgotten")

            method = _shared

        assert await _Enforced().method() is True
        with pytest.raises(AssertionError, match="was called on a _Forgotten"):
            await _Forgotten().method()

    @pytest.mark.asyncio
    async def test_a_guarded_classmethod_runs_on_an_enforced_class(self):
        """A classmethod is called with the class itself, which still belongs to the enforcing class."""

        @enforce_lock_discipline
        class _WithClassmethod:
            context_lock = ContextLock("classmethod")

            @lock_exempt
            def __init__(self) -> None: ...

            @classmethod
            @with_lock
            async def method(cls) -> bool:
                return cls.context_lock.held_in_current_context

        assert await _WithClassmethod.method() is True

    @pytest.mark.asyncio
    async def test_a_guarded_method_wrapping_a_bound_method_is_accepted(self):
        """Walking __wrapped__ onto a bound method must not fail the class definition."""

        class _Source:
            @lock_exempt
            async def implementation(self) -> str:
                return "ok"

        bound = _Source().implementation

        @functools.wraps(bound)
        async def _outer(self) -> str:
            return await bound()

        @enforce_lock_discipline
        class _Delegating:
            @lock_exempt
            def __init__(self) -> None:
                self.context_lock = ContextLock("delegating")

            method = _outer

        assert await _Delegating().method() == "ok"

    def test_lock_exempt_leaves_the_function_behaviour_untouched(self):
        """The exemption is a marker only; it must not wrap or alter the call."""

        @lock_exempt
        def add(left: int, right: int) -> int:
            return left + right

        assert add(1, 2) == 3
