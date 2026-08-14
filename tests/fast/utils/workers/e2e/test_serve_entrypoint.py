import contextlib
import os
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from tests.fast.utils.workers.e2e.e2e_worker import WORKER_FACTORY_ERROR
from tests.fast.utils.workers.e2e.env_var_hooks import ENV_VAR_FN_FAILURE_MESSAGE, IMPORTED_MODULES_ENV_VAR
from tests.fast.utils.workers.e2e.harness import (
    READY_TIMEOUT_SECONDS,
    REPO_ROOT,
    WORKER_PATH,
    ServerProcess,
    port_is_refused,
    reserve_port,
    wait_until_serving,
)
from tests.fast.utils.workers.import_probe import unexpected_light_entrypoint_imports

SMOKE_MODULE = "tests.fast.utils.workers.e2e.env_var_hooks"
SMOKE_ENV_FN_PATH = f"{SMOKE_MODULE}.compute_env_vars"
SMOKE_RAISING_ENV_FN_PATH = f"{SMOKE_MODULE}.raise_env_var_error"
RAISING_WORKER_PATH = "tests.fast.utils.workers.e2e.e2e_worker.make_raising_worker"
EXIT_TIMEOUT_SECONDS = 60.0


@pytest.fixture
def spawn_with_env_var_fn(state_dir: Path, tmp_path: Path) -> Iterator[Callable[..., ServerProcess]]:
    started: list[ServerProcess] = []

    def start(env_var_fn: str, *, worker_path: str = WORKER_PATH) -> ServerProcess:
        port = reserve_port()
        log_path = tmp_path / f"env-var-fn-server-{len(started)}.log"

        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"

        argv = [sys.executable, "-m", "miles.utils.workers.serving.serve", "--worker", worker_path]
        argv += ["--env-var-fn", env_var_fn, "--host", "127.0.0.1", "--port", str(port)]
        argv += ["--", "--state-dir", str(state_dir)]

        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                argv, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
            )

        server = ServerProcess(port=port, process=process, log_path=log_path)
        started.append(server)
        return server

    yield start

    for server in started:
        server.stop()
        server.kill()


class TestExecChain:
    async def test_the_served_process_is_the_spawned_one(self, handle, server):
        """execve keeps the pid, so terminating the spawned process really stops the server."""
        assert await handle.report_pid() == server.process.pid

    async def test_worker_argv_reaches_the_factory(self, handle):
        """Everything after -- is handed to the worker factory."""
        argv = await handle.report_argv()
        assert "--state-dir" in argv

    async def test_worker_argv_keeps_its_own_separator(self, spawn, make_handle):
        """Only the first -- splits, so worker argv may contain further separators."""
        server = spawn(worker_argv=["--flag", "--", "--inner"])
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        argv = await handle.report_argv()
        assert argv[-3:] == ["--flag", "--", "--inner"]

    async def test_env_var_hook_receives_worker_argv(self, handle):
        """The env-var hook is called with the worker argv, not the entrypoint argv."""
        recorded = await handle.report_env(name="MILES_E2E_ARGV")
        assert "--state-dir" in recorded

    async def test_only_allowlisted_modules_are_imported_before_the_hook(self, spawn_with_env_var_fn, make_handle):
        """When the hook runs, the light entrypoint has imported no top-level module outside the allowlist."""
        server = spawn_with_env_var_fn(SMOKE_ENV_FN_PATH)
        wait_until_serving(server)
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        reported = await handle.report_env(name=IMPORTED_MODULES_ENV_VAR)
        assert unexpected_light_entrypoint_imports(reported) == []

    async def test_parent_environment_is_inherited(self, spawn, make_handle):
        """Environment from the launcher reaches the worker."""
        server = spawn(extra_env={"MILES_E2E_MARKER": "inherited"})
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        assert await handle.report_env(name="MILES_E2E_MARKER") == "inherited"

    async def test_env_var_hook_overrides_an_inherited_value(self, spawn, make_handle):
        """A computed variable replaces the same-named value the launcher exported."""
        server = spawn(extra_env={"MILES_E2E_ARGV": "from-parent"})
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        recorded = await handle.report_env(name="MILES_E2E_ARGV")
        assert recorded != "from-parent"
        assert "--state-dir" in recorded

    async def test_env_var_hook_is_optional(self, spawn, make_handle):
        """Serving without the hook still works."""
        server = spawn(env_var_fn=False)
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        assert await handle.demo_sync(a=1, b=1) == 2
        assert await handle.report_env(name="MILES_E2E_ARGV") is None


class TestStartupFailures:
    async def test_unknown_worker_path_fails_fast(self, spawn):
        """A worker path that cannot be imported exits instead of serving."""
        server = spawn(worker_path="no.such.module.make_worker", wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

    async def test_missing_worker_argument_is_a_usage_error(self, spawn):
        """argparse rejects a missing --worker with its usage exit code."""
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [sys.executable, "-m", "miles.utils.workers.serving.serve", "--host", "127.0.0.1"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            timeout=60,
        )

        assert result.returncode == 2
        assert b"usage" in result.stderr.lower()

    async def test_port_conflict_fails_fast(self, spawn, server):
        """A second server on a taken port exits without disturbing the first."""
        conflicting = spawn(port=server.port, wait=False)
        assert conflicting.wait(timeout=30.0) not in (None, 0)
        assert server.is_running()

    @pytest.mark.parametrize("bad_path", ["no_colon_module", "miles.utils.workers.serving.serve.no_such_attr"])
    async def test_bad_factory_paths_fail_fast(self, spawn, bad_path):
        """Malformed or missing factory paths exit rather than serving a broken worker."""
        server = spawn(worker_path=bad_path, wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)

    async def test_unknown_env_var_fn_module_fails_fast(self, spawn_with_env_var_fn):
        """An env-var hook whose module cannot be imported exits instead of serving."""
        server = spawn_with_env_var_fn("no.such.module.compute_env_vars")
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)
        assert "ModuleNotFoundError" in server.logs()

    async def test_missing_env_var_fn_attribute_fails_fast(self, spawn_with_env_var_fn):
        """An env-var hook naming an attribute the module lacks exits instead of serving."""
        server = spawn_with_env_var_fn(f"{SMOKE_MODULE}.no_such_attr")
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "AttributeError" in logs
        assert "no_such_attr" in logs

    async def test_raising_env_var_fn_fails_fast(self, spawn_with_env_var_fn):
        """An env-var hook that raises when called exits instead of serving, and reports its own error."""
        server = spawn_with_env_var_fn(SMOKE_RAISING_ENV_FN_PATH)
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "RuntimeError" in logs
        assert ENV_VAR_FN_FAILURE_MESSAGE in logs


class TestWorkerFactoryFailure:
    def test_raising_worker_factory_fails_before_binding_the_port(self, spawn) -> None:
        """A worker factory that raises fails startup before the port is bound and reports its own error."""
        server = spawn(wait=False, worker_path=RAISING_WORKER_PATH)
        exit_code = server.wait(EXIT_TIMEOUT_SECONDS)

        assert exit_code is not None and exit_code != 0, f"server did not exit:\n{server.logs()}"
        assert (
            WORKER_FACTORY_ERROR in server.logs()
        ), f"startup failed before reaching the worker factory:\n{server.logs()}"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", server.port))


class TestPortBinding:
    async def test_server_is_unreachable_on_non_loopback_addresses(self, server):
        """Binding 127.0.0.1 keeps the port reachable on loopback and refused on the machine's other address."""
        address = _non_loopback_ipv4_address()
        if address is None:
            pytest.skip("no non-loopback ipv4 address on this machine")

        assert not port_is_refused(server.port)
        assert _connection_is_refused(address, server.port)


def _non_loopback_ipv4_address() -> str | None:
    candidates: list[str] = []

    with contextlib.suppress(OSError):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 53))
            candidates.append(probe.getsockname()[0])

    with contextlib.suppress(OSError):
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            candidates.append(info[4][0])

    for address in candidates:
        if not address.startswith("127.") and address != "0.0.0.0" and _is_local_address(address):
            return address
    return None


def _is_local_address(address: str) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((address, 0))
        except OSError:
            return False
    return True


def _connection_is_refused(address: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2.0)
        try:
            return sock.connect_ex((address, port)) != 0
        except OSError:
            return True
