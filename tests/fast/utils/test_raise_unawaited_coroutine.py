"""Tests for configure_raise_unawaited_coroutine."""

import asyncio
import gc
import warnings

import pytest

from miles.utils.logging_utils import configure_raise_unawaited_coroutine


async def _dummy_coroutine():
    return 42


@pytest.fixture(autouse=True)
def _setup_warning_filter():
    """Activate the filter before each test, restore original filters after."""
    with warnings.catch_warnings():
        configure_raise_unawaited_coroutine()
        yield


class TestUnawaitedCoroutineRaises:
    def test_unawaited_coroutine_raises_runtime_error(self):
        """Calling an async function without await should raise RuntimeWarning-as-error."""
        with pytest.raises(RuntimeWarning, match="coroutine .* was never awaited"):
            _dummy_coroutine()  # not awaited
            gc.collect()  # force GC to trigger the warning

    def test_unawaited_coroutine_in_sync_context(self):
        """Same as above but in a more realistic pattern: assigning to a variable."""
        with pytest.raises(RuntimeWarning, match="coroutine .* was never awaited"):
            _coro = _dummy_coroutine()  # noqa: F841
            del _coro
            gc.collect()


class TestAwaitedCoroutineDoesNotRaise:
    def test_properly_awaited_coroutine_no_error(self):
        """A properly awaited coroutine should not raise."""
        result = asyncio.get_event_loop().run_until_complete(_dummy_coroutine())
        assert result == 42

    @pytest.mark.asyncio
    async def test_awaited_in_async_context(self):
        """Awaiting in an async test should not raise."""
        result = await _dummy_coroutine()
        assert result == 42

    @pytest.mark.asyncio
    async def test_gathered_coroutines_no_error(self):
        """Coroutines passed to asyncio.gather should not raise."""
        results = await asyncio.gather(_dummy_coroutine(), _dummy_coroutine())
        assert results == [42, 42]

    @pytest.mark.asyncio
    async def test_create_task_then_await_no_error(self):
        """create_task + await pattern should not raise."""
        task = asyncio.create_task(_dummy_coroutine())
        result = await task
        assert result == 42


class TestOtherWarningsUnaffected:
    def test_unrelated_runtime_warning_not_raised(self):
        """Non-coroutine RuntimeWarnings should still be warnings, not errors."""
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            configure_raise_unawaited_coroutine()
            # This should NOT raise — it doesn't match the coroutine pattern
            with pytest.warns(RuntimeWarning, match="test warning"):
                warnings.warn("test warning", RuntimeWarning)

    def test_deprecation_warning_not_raised(self):
        """DeprecationWarning should not be affected."""
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            configure_raise_unawaited_coroutine()
            with pytest.warns(DeprecationWarning):
                warnings.warn("old stuff", DeprecationWarning)
