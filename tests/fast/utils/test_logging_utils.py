"""Tests for configure_strict_async_warnings."""

import asyncio
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


class TestUnawaitedCoroutineFilterCatchesWarning:
    def test_matching_warning_becomes_error(self):
        """The filter turns coroutine-never-awaited warnings into errors."""
        with pytest.raises(RuntimeWarning, match="coroutine .* was never awaited"):
            warnings.warn("coroutine 'foo' was never awaited", RuntimeWarning, stacklevel=2)

    def test_partial_match_also_caught(self):
        with pytest.raises(RuntimeWarning, match="coroutine .* was never awaited"):
            warnings.warn(
                "coroutine 'MyClass.some_method' was never awaited", RuntimeWarning, stacklevel=2
            )


class TestDestroyedTaskFilterCatchesWarning:
    def test_matching_warning_becomes_error(self):
        """The filter turns destroyed-pending-task warnings into errors."""
        with pytest.raises(RuntimeWarning, match="Task.*was destroyed but it is pending"):
            warnings.warn("Task was destroyed but it is pending!", RuntimeWarning, stacklevel=2)


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
                warnings.warn("test warning", RuntimeWarning, stacklevel=2)

    def test_deprecation_warning_not_raised(self):
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            configure_strict_async_warnings()
            with pytest.warns(DeprecationWarning):
                warnings.warn("old stuff", DeprecationWarning, stacklevel=2)
