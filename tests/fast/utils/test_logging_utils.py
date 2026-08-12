"""Tests for configure_strict_async_warnings."""

import asyncio
import importlib
import inspect
import pkgutil
import subprocess
import sys
import textwrap
import warnings

import pytest

from miles.utils.logging_utils import configure_strict_async_warnings
from miles.utils.workers.ray_worker_manager import RayWorkerManager

SPECS_PACKAGE = "miles.ray.specs"


def _worker_class_paths() -> list[str]:
    """Every class a spec can name as its worker: the *_WORKER_CLASS constants, plus the mappings
    a spec picks a backend's rank class out of."""
    paths: set[str] = set()
    package = importlib.import_module(SPECS_PACKAGE)
    for module in pkgutil.iter_modules(package.__path__):
        for name, value in vars(importlib.import_module(f"{SPECS_PACKAGE}.{module.name}")).items():
            if name.endswith("_WORKER_CLASS"):
                paths.add(value)
            elif name.endswith("_CLASSES") and isinstance(value, dict):
                paths.update(value.values())

    assert paths, f"no worker classes found in {SPECS_PACKAGE}"
    return sorted(paths)


def _load_class(path: str):
    module_path, _, class_name = path.rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def _class_source(klass: type) -> str | None:
    try:
        return inspect.getsource(klass)
    except (OSError, TypeError):
        return None


def _configures_logger(klass: type) -> bool:
    sources = [source for base in klass.__mro__ if (source := _class_source(base)) is not None]
    return any("configure_logger(" in source for source in sources)


class TestServedWorkerLogging:
    @pytest.mark.parametrize("worker_class_path", _worker_class_paths())
    def test_worker_configures_its_logger(self, worker_class_path: str) -> None:
        """Every served worker owns a process, and configure_logger is where that process names
        itself and opens its event log."""
        assert _configures_logger(
            _load_class(worker_class_path)
        ), f"{worker_class_path} runs as its own process but never calls configure_logger"

    @pytest.mark.parametrize(
        ("module_path", "component"),
        [
            ("miles.ray.rollout.inference_controller", "inference_controller"),
            ("miles.ray.rollout.rollout_executor", "rollout_executor"),
            ("miles.ray.multi_lora.controller", "multi_lora_controller"),
            ("miles.utils.workers.ray_worker_manager", "worker_manager"),
        ],
    )
    def test_each_process_names_itself_as_the_component_it_is(self, module_path: str, component: str) -> None:
        """One SimpleProcessIdentity now serves them all, so only the argument tells their events apart."""
        source = inspect.getsource(importlib.import_module(module_path))

        assert f'SimpleProcessIdentity(component="{component}")' in source

    def test_the_worker_manager_configures_its_logger(self) -> None:
        """The ray worker manager is its own actor process, but no spec names it as a worker class."""
        assert _configures_logger(RayWorkerManager)


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
        timeout=60,
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
