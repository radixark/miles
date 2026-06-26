"""Tests for configure_strict_async_warnings and the Ray actor log-prefix helpers."""

import asyncio
import io
import subprocess
import sys
import textwrap
import threading
import warnings

import pytest

import miles.utils.logging_utils as logging_utils
from miles.utils.logging_utils import (
    LinePrefixingStream,
    cached_dist_rank,
    configure_strict_async_warnings,
    get_ray_friendly_repr,
    install_ray_actor_log_prefix,
)


async def _dummy_coroutine():
    return 42


@pytest.fixture(autouse=True)
def _setup_warning_filter():
    """Activate the filter before each test, restore original filters after."""
    original_hook = sys.unraisablehook
    with warnings.catch_warnings():
        configure_strict_async_warnings()
        yield
    sys.unraisablehook = original_hook


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
        result = asyncio.run(_dummy_coroutine())
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


class TestLinePrefixingStream:
    @staticmethod
    def _wrap(sink: io.StringIO) -> LinePrefixingStream:
        return LinePrefixingStream(sink, lambda: "TAG")

    def test_single_line_print_gets_one_prefix(self):
        """A `print("hello")` (text then newline) yields exactly one leading prefix."""
        sink = io.StringIO()
        stream = self._wrap(sink)
        stream.write("hello")
        stream.write("\n")
        assert sink.getvalue() == "[TAG] hello\n"

    def test_multiline_write_prefixes_each_line(self):
        """Each line in a multi-line write gets its own prefix."""
        sink = io.StringIO()
        self._wrap(sink).write("a\nb\n")
        assert sink.getvalue() == "[TAG] a\n[TAG] b\n"

    def test_partial_writes_within_a_line_share_one_prefix(self):
        """Writes that build up a single line are prefixed only at the line start."""
        sink = io.StringIO()
        stream = self._wrap(sink)
        stream.write("ab")
        stream.write("cd\n")
        assert sink.getvalue() == "[TAG] abcd\n"

    def test_blank_line_is_not_prefixed(self):
        """An empty line stays empty rather than emitting a bare prefix."""
        sink = io.StringIO()
        self._wrap(sink).write("\n")
        assert sink.getvalue() == "\n"

    def test_delegates_unknown_attributes_to_underlying(self):
        """Attribute access falls through to the wrapped stream (e.g. getvalue)."""
        sink = io.StringIO()
        assert self._wrap(sink).getvalue() == ""

    def test_writelines_prefixes_each_line(self):
        """writelines is routed through write so its lines are prefixed too."""
        sink = io.StringIO()
        self._wrap(sink).writelines(["a\n", "b\n"])
        assert sink.getvalue() == "[TAG] a\n[TAG] b\n"

    def test_concurrent_writes_do_not_interleave(self):
        """Each whole-line write stays intact under concurrent writers (RLock)."""
        sink = io.StringIO()
        stream = self._wrap(sink)
        num_threads, lines_per_thread = 8, 200
        payloads = [f"t{tid}-{j}" for tid in range(num_threads) for j in range(lines_per_thread)]

        def writer(tid: int) -> None:
            for j in range(lines_per_thread):
                stream.write(f"t{tid}-{j}\n")

        threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        lines = sink.getvalue().splitlines()
        assert sorted(lines) == sorted(f"[TAG] {payload}" for payload in payloads)


class TestInstallRayActorLogPrefix:
    @pytest.fixture(autouse=True)
    def _restore_std_streams(self):
        original_stdout, original_stderr = sys.stdout, sys.stderr
        yield
        sys.stdout, sys.stderr = original_stdout, original_stderr

    def test_wraps_stdout_and_stderr_idempotently(self):
        """Installing wraps both streams once and is a no-op on repeat calls."""
        install_ray_actor_log_prefix()
        assert isinstance(sys.stdout, LinePrefixingStream)
        assert isinstance(sys.stderr, LinePrefixingStream)

        wrapped_stdout, wrapped_stderr = sys.stdout, sys.stderr
        install_ray_actor_log_prefix()
        assert sys.stdout is wrapped_stdout
        assert sys.stderr is wrapped_stderr


class TestRankResolution:
    @pytest.fixture(autouse=True)
    def _reset_rank_cache(self):
        original = logging_utils._cached_dist_rank
        logging_utils._cached_dist_rank = None
        yield
        logging_utils._cached_dist_rank = original

    def test_falls_back_to_rank_env_when_dist_uninitialized(self, monkeypatch):
        """Before the process group is up, the rank comes from the RANK env var."""
        monkeypatch.setenv("RANK", "3")
        assert cached_dist_rank() == 3

    def test_returns_none_without_dist_or_env(self, monkeypatch):
        """With neither a process group nor RANK set, the rank is unknown."""
        monkeypatch.delenv("RANK", raising=False)
        assert cached_dist_rank() is None

    def test_friendly_repr_contains_rank_and_time(self, monkeypatch):
        """The per-line tag carries the rank and an HH:MM:SS.mmm timestamp."""
        monkeypatch.setenv("RANK", "5")
        tag = get_ray_friendly_repr()
        assert "rank=5" in tag
        assert tag.count(":") == 2 and "." in tag
