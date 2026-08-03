import asyncio

import pytest

from miles.utils.context_lock import ContextLock, with_lock


class _Guarded:
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

        class _Nested:
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

        class _NoLock:
            @with_lock
            async def method(self) -> None:
                pass

        with pytest.raises(AttributeError, match="context_lock"):
            await _NoLock().method()

    @pytest.mark.asyncio
    async def test_a_lock_attribute_of_the_wrong_type_is_reported(self):
        """Handing over a bare asyncio.Lock loses the held-by tracking, so it is refused."""

        class _WrongLock:
            def __init__(self) -> None:
                self.context_lock = asyncio.Lock()

            @with_lock
            async def method(self) -> None:
                pass

        with pytest.raises(AssertionError, match="must be a ContextLock"):
            await _WrongLock().method()


async def _read_held(lock: ContextLock) -> bool:
    return lock.held_in_current_context
