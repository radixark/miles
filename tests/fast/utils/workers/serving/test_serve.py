import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from tests.fast.utils.workers.import_probe import unexpected_light_entrypoint_imports
from tests.fast.utils.workers.serving.serve_smoke_worker import (
    IMPORTED_MODULES_ENV_VAR,
    POOL_ID,
    RPC_PORT_FLAG,
    SMOKE_EXTRA_ENV_VAR,
    SmokeWorker,
)

from miles.utils.http_utils import find_available_port
from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.serving import serve as serve_module
from miles.utils.workers.serving.utils import split_worker_argv
from miles.utils.workers.worker_handle import WorkerUnreachableError

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SPECS_PATH = "tests.fast.utils.workers.serving.serve_smoke_worker.compute_specs"


class TestSplitWorkerArgv:
    def test_splits_on_the_separator(self):
        """Everything after -- is worker argv, everything before belongs to the entrypoint."""
        assert split_worker_argv(["--pool-id", "p", "--", "--greeting", "hi"]) == (
            ["--pool-id", "p"],
            ["--greeting", "hi"],
        )

    def test_no_separator_means_no_worker_argv(self):
        """Without a separator the whole argv belongs to the entrypoint."""
        assert split_worker_argv(["--pool-id", "p"]) == (["--pool-id", "p"], [])

    def test_later_separators_stay_with_the_worker(self):
        """Only the first separator splits, so worker argv may contain its own --."""
        assert split_worker_argv(["--", "a", "--", "b"]) == ([], ["a", "--", "b"])


class TestOuterServeForwarding:
    def test_forwards_its_own_argv_and_the_env_the_spec_computed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """serve exists to put the spec's env on the exec'd image; anything it swallowed would never arrive."""
        captured: dict[str, object] = {}

        def fake_execve(path: str, argv: list[str], env: dict[str, str]) -> None:
            captured.update(path=path, argv=argv, env=env)

        monkeypatch.setattr(serve_module.os, "execve", fake_execve)
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        own_argv = ["--specs", _SPECS_PATH, "--pool-id", POOL_ID]
        worker_argv = [RPC_PORT_FLAG, "9000", "--flag", "value"]
        monkeypatch.setattr(sys, "argv", ["serve.py", *own_argv, "--", *worker_argv])

        serve_module.main()

        assert captured["argv"] == [
            sys.executable,
            "-m",
            "miles.utils.workers.serving.serve_inner",
            *own_argv,
            "--",
            *worker_argv,
        ]
        assert captured["env"]["MILES_SERVE_SMOKE_ENV"] == ",".join(worker_argv)
        assert captured["env"]["MILES_SERVE_SMOKE_POOL_ID"] == POOL_ID

    def test_refuses_a_spec_that_overwrites_the_identity_the_platform_gave_the_pod(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--train-env-vars is unrestricted json, and one of these keys makes every rank claim worker zero."""
        monkeypatch.setattr(serve_module.os, "execve", _refuse_exec)
        monkeypatch.setenv(CELL_INDEX_ENV_VAR, "0")
        monkeypatch.setenv(SMOKE_EXTRA_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR)
        own_argv = ["--specs", _SPECS_PATH, "--pool-id", POOL_ID]
        monkeypatch.setattr(sys, "argv", ["serve.py", *own_argv, "--", RPC_PORT_FLAG, "9000"])

        with pytest.raises(AssertionError, match=SUBPROCESS_INDEX_ENV_VAR):
            serve_module.main()


def _refuse_exec(path: str, argv: list[str], env: dict[str, str]) -> None:
    raise AssertionError("a spec that overwrites the pod's identity must not reach exec")

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
    env[CELL_INDEX_ENV_VAR] = "0"

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "miles.utils.workers.serving.serve",
            "--specs",
            _SPECS_PATH,
            "--pool-id",
            POOL_ID,
            "--",
            RPC_PORT_FLAG,
            str(port),
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
                assert (await handle.report_argv())[-2:] == ["--greeting", "hello"]
                assert (await handle.report_env(name="MILES_SERVE_SMOKE_ENV")).endswith("--greeting,hello")
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
