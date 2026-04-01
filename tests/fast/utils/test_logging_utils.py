"""Tests for configure_strict_async_warnings."""

import asyncio
import gc
import warnings

import pytest

from miles.utils.logging_utils import configure_strict_async_warnings


async def _dummy_coroutine():
    return 42


@pytest.fixture(autouse=True)
def _setup_warning_filter():
    """Activate the filter before each test, restore original filters after."""
    with warnings.catch_warnings():
        configure_strict_async_warnings()
        yield


class TestUnawaitedCoroutineRaises:
    def test_unawaited_coroutine_raises(self):
        """Calling an async function without await should raise."""
        with pytest.raises(RuntimeWarning, match="coroutine .* was never awaited"):
            _dummy_coroutine()  # not awaited
            gc.collect()

    def test_unawaited_coroutine_del_and_gc(self):
        """Assigning then deleting an unawaited coroutine should raise."""
        with pytest.raises(RuntimeWarning, match="coroutine .* was never awaited"):
            _coro = _dummy_coroutine()  # noqa: F841
            del _coro
            gc.collect()


class TestDestroyedPendingTaskRaises:
    @pytest.mark.asyncio
    async def test_task_lost_reference_raises(self):
        """A task whose reference is lost while still pending should raise."""

        async def slow():
            await asyncio.sleep(100)

        with pytest.raises(RuntimeWarning, match="Task.*was destroyed but it is pending"):
            _task = asyncio.create_task(slow())  # noqa: F841
            del _task
            gc.collect()

    @pytest.mark.asyncio
    async def test_task_completed_before_gc_no_error(self):
        """A task that completes before losing reference should NOT raise."""
        task = asyncio.create_task(_dummy_coroutine())
        await task
        del task
        gc.collect()

    @pytest.mark.asyncio
    async def test_task_cancelled_before_gc_no_error(self):
        """A task that is cancelled before losing reference should NOT raise."""

        async def slow():
            await asyncio.sleep(100)

        task = asyncio.create_task(slow())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        del task
        gc.collect()


class TestCorrectUsageNoError:
    def test_properly_awaited_coroutine(self):
        result = asyncio.get_event_loop().run_until_complete(_dummy_coroutine())
        assert result == 42

    @pytest.mark.asyncio
    async def test_awaited_in_async_context(self):
        result = await _dummy_coroutine()
        assert result == 42

    @pytest.mark.asyncio
    async def test_gathered_coroutines(self):
        results = await asyncio.gather(_dummy_coroutine(), _dummy_coroutine())
        assert results == [42, 42]

    @pytest.mark.asyncio
    async def test_create_task_then_await(self):
        task = asyncio.create_task(_dummy_coroutine())
        result = await task
        assert result == 42

    @pytest.mark.asyncio
    async def test_eager_create_task(self):
        from miles.utils.async_utils import eager_create_task

        task = await eager_create_task(_dummy_coroutine())
        result = await task
        assert result == 42


class TestOtherWarningsUnaffected:
    def test_unrelated_runtime_warning_not_raised(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            configure_strict_async_warnings()
            with pytest.warns(RuntimeWarning, match="test warning"):
                warnings.warn("test warning", RuntimeWarning)

    def test_deprecation_warning_not_raised(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            configure_strict_async_warnings()
            with pytest.warns(DeprecationWarning):
                warnings.warn("old stuff", DeprecationWarning)
