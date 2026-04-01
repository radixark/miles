"""Tests for configure_strict_async_warnings."""

import asyncio
import subprocess
import sys
import textwrap
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


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=10,
    )


class TestUnawaitedCoroutineCrashesProcess:
    def test_unawaited_coroutine_exits_with_code_1(self):
        result = _run_snippet(
            """
            import gc
            from miles.utils.logging_utils import configure_strict_async_warnings
            configure_strict_async_warnings()

            async def foo(): pass
            foo()
            gc.collect()
            print("should not reach here")
        """
        )
        assert result.returncode == 1
        assert "should not reach here" not in result.stdout
        assert "Fatal async misuse" in result.stderr

    def test_unawaited_coroutine_del_exits_with_code_1(self):
        result = _run_snippet(
            """
            import gc
            from miles.utils.logging_utils import configure_strict_async_warnings
            configure_strict_async_warnings()

            async def foo(): pass
            c = foo()
            del c
            gc.collect()
            print("should not reach here")
        """
        )
        assert result.returncode == 1
        assert "should not reach here" not in result.stdout
        assert "coroutine" in result.stderr

    def test_awaited_coroutine_no_crash(self):
        result = _run_snippet(
            """
            import asyncio
            from miles.utils.logging_utils import configure_strict_async_warnings
            configure_strict_async_warnings()

            async def foo(): return 42
            print(asyncio.run(foo()))
        """
        )
        assert result.returncode == 0
        assert "42" in result.stdout


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
