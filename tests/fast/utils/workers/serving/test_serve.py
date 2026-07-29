import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from tests.fast.utils.workers.import_probe import unexpected_light_entrypoint_imports
from tests.fast.utils.workers.serving.serve_smoke_worker import IMPORTED_MODULES_ENV_VAR, SmokeWorker

from miles.utils.http_utils import find_available_port
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.serving import serve as serve_module
from miles.utils.workers.serving.utils import split_worker_argv
from miles.utils.workers.worker_handle import WorkerUnreachableError

_REPO_ROOT = Path(__file__).resolve().parents[5]
_WORKER_PATH = "tests.fast.utils.workers.serving.serve_smoke_worker.make_worker"
_ENV_FN_PATH = "tests.fast.utils.workers.serving.serve_smoke_worker.compute_env_vars"


class TestSplitWorkerArgv:
    def test_splits_on_the_separator(self):
        """Everything after -- is worker argv, everything before belongs to the entrypoint."""
        assert split_worker_argv(["--worker", "m:f", "--", "--greeting", "hi"]) == (
            ["--worker", "m:f"],
            ["--greeting", "hi"],
        )

    def test_no_separator_means_no_worker_argv(self):
        """Without a separator the whole argv belongs to the entrypoint."""
        assert split_worker_argv(["--worker", "m:f"]) == (["--worker", "m:f"], [])

    def test_later_separators_stay_with_the_worker(self):
        """Only the first separator splits, so worker argv may contain its own --."""
        assert split_worker_argv(["--", "a", "--", "b"]) == ([], ["a", "--", "b"])


class TestOuterServeForwarding:
    def test_only_env_hook_is_consumed_before_exec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The outer serve layer forwards all non-hook own arguments verbatim."""
        captured: dict[str, object] = {}

        def fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
            captured.update(path=path, argv=argv, env=env)

        monkeypatch.setattr(serve_module.os, "execve", fake_execve)
        monkeypatch.setattr(
            serve_module,
            "load_function",
            lambda path: lambda worker_argv: {"MILES_TEST_ENV": ",".join(worker_argv)},
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "serve.py",
                "--worker",
                "package.worker",
                "--host",
                "127.0.0.1",
                "--env-var-fn",
                "package.env",
                "--port",
                "9000",
                "--",
                "--flag",
                "value",
            ],
        )

        serve_module.main()

        assert captured["argv"] == [
            sys.executable,
            "-m",
            "miles.utils.workers.serving.serve_inner",
            "--worker",
            "package.worker",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--",
            "--flag",
            "value",
        ]
        assert captured["env"]["MILES_TEST_ENV"] == "--flag,value"

    def test_env_var_hook_overrides_same_named_parent_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A computed variable wins over a same-named variable inherited from the parent environment."""
        captured: dict[str, object] = {}

        def fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
            captured.update(path=path, argv=argv, env=env)

        monkeypatch.setenv("MILES_TEST_ENV", "from-parent")
        monkeypatch.setenv("MILES_TEST_UNTOUCHED", "kept")
        monkeypatch.setattr(serve_module.os, "execve", fake_execve)
        monkeypatch.setattr(
            serve_module,
            "load_function",
            lambda path: lambda worker_argv: {"MILES_TEST_ENV": ",".join(worker_argv)},
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "serve.py",
                "--worker",
                "package.worker",
                "--env-var-fn",
                "package.env",
                "--",
                "--flag",
                "value",
            ],
        )

        serve_module.main()

        assert captured["env"]["MILES_TEST_ENV"] == "--flag,value"
        assert captured["env"]["MILES_TEST_UNTOUCHED"] == "kept"


def _spawn_serve(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "miles.utils.workers.serving.serve",
            "--worker",
            _WORKER_PATH,
            "--env-var-fn",
            _ENV_FN_PATH,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--",
            "--greeting",
            "hello",
        ],
        cwd=_REPO_ROOT,
        env=env,
    )


async def _wait_ready_or_die(handle, process: subprocess.Popen, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        assert process.poll() is None, f"serve subprocess exited early with code {process.returncode}"
        with contextlib.suppress(WorkerUnreachableError):
            await handle.wait_ready(timeout=2.0)
            return
    raise AssertionError("serve subprocess never became ready")


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


class TestServeEndToEnd:
    async def test_serve_call_argv_and_env(self):
        """serve.py boots a worker subprocess that answers typed calls with argv and env applied."""
        port = find_available_port(20000 + os.getpid() % 10000)
        process = _spawn_serve(port)

        async with httpx.AsyncClient(trust_env=False) as client:
            handle = RpcWorkerHandle(
                SmokeWorker,
                server_url=f"http://127.0.0.1:{port}",
                http_client=client,
            )
            try:
                await _wait_ready_or_die(handle, process)
                assert await handle.demo_sync(a=3, b=4) == 7
                assert await handle.report_argv() == ["--greeting", "hello"]
                assert await handle.report_env(name="MILES_SERVE_SMOKE_ENV") == "--greeting,hello"
                reported = await handle.report_env(name=IMPORTED_MODULES_ENV_VAR)
                assert unexpected_light_entrypoint_imports(reported) == []
            finally:
                _stop(process)

    async def test_restart_detected_by_stable_boot_uuid_client(self):
        """A stable-boot-uuid client notices when the serve subprocess is restarted."""
        port = find_available_port(21000 + os.getpid() % 10000)
        first_process = _spawn_serve(port)
        second_process: subprocess.Popen | None = None

        async with httpx.AsyncClient(trust_env=False) as client:
            handle = RpcWorkerHandle(
                SmokeWorker,
                server_url=f"http://127.0.0.1:{port}",
                require_stable_boot_uuid=True,
                http_client=client,
            )
            try:
                await _wait_ready_or_die(handle, first_process)
                assert await handle.demo_sync(a=1, b=1) == 2

                _stop(first_process)
                second_process = _spawn_serve(port)
                fresh_handle = RpcWorkerHandle(
                    SmokeWorker,
                    server_url=f"http://127.0.0.1:{port}",
                    http_client=client,
                )
                await _wait_ready_or_die(fresh_handle, second_process)

                with pytest.raises(ServerRestartedError):
                    await handle.demo_sync(a=1, b=1)
            finally:
                if second_process is not None:
                    _stop(second_process)
                _stop(first_process)
