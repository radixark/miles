import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from tests.fast.utils.workers.e2e.env_var_hooks import ENV_VAR_FN_FAILURE_MESSAGE, IMPORTED_MODULES_ENV_VAR
from tests.fast.utils.workers.e2e.harness import (
    POOL_ID,
    READY_TIMEOUT_SECONDS,
    REPO_ROOT,
    RPC_PORT_FLAG,
    ServerProcess,
    port_is_refused,
    reserve_port,
    wait_until_serving,
)
from tests.fast.utils.workers.import_probe import unexpected_light_entrypoint_imports

from miles.utils.workers.env_vars import CELL_INDEX_ENV_VAR

SMOKE_MODULE = "tests.fast.utils.workers.e2e.env_var_hooks"
SMOKE_SPECS_PATH = f"{SMOKE_MODULE}.compute_specs"
SMOKE_RAISING_SPECS_PATH = f"{SMOKE_MODULE}.compute_failing_specs"


@pytest.fixture
def spawn_with_specs(state_dir: Path, tmp_path: Path) -> Iterator[Callable[..., ServerProcess]]:
    started: list[ServerProcess] = []

    def start(specs_path: str) -> ServerProcess:
        port = reserve_port()
        log_path = tmp_path / f"specs-server-{len(started)}.log"

        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"
        env[CELL_INDEX_ENV_VAR] = "0"

        argv = [sys.executable, "-m", "miles.utils.workers.serving.serve"]
        argv += ["--specs", specs_path, "--pool-id", POOL_ID]
        argv += ["--", "--state-dir", str(state_dir), RPC_PORT_FLAG, str(port)]

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

    async def test_the_spec_computes_its_env_from_the_worker_argv(self, handle):
        """The spec is rebuilt from the run's own argv, not from the entrypoint's."""
        recorded = await handle.report_env(name="MILES_E2E_ARGV")
        assert "--state-dir" in recorded

    async def test_no_heavy_runtime_is_imported_before_the_exec(self, spawn_with_specs, make_handle):
        """LD_PRELOAD applies to the exec'd image, so a runtime loaded before it would miss the spec's env."""
        server = spawn_with_specs(SMOKE_SPECS_PATH)
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


class TestStartupFailures:
    async def test_unknown_specs_path_fails_fast(self, spawn):
        """A spec table that cannot be imported exits instead of serving."""
        server = spawn(specs_path="no.such.module.compute_specs", wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

    async def test_missing_specs_argument_is_a_usage_error(self, spawn):
        """argparse rejects a missing --specs with its usage exit code."""
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [sys.executable, "-m", "miles.utils.workers.serving.serve", "--pool-id", POOL_ID],
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
    async def test_bad_specs_paths_fail_fast(self, spawn, bad_path):
        """Malformed or missing spec-table paths exit rather than serving a broken worker."""
        server = spawn(specs_path=bad_path, wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)

    async def test_unknown_specs_module_fails_fast(self, spawn_with_specs):
        """A spec table whose module cannot be imported exits instead of serving."""
        server = spawn_with_specs("no.such.module.compute_specs")
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)
        assert "ModuleNotFoundError" in server.logs()

    async def test_missing_specs_attribute_fails_fast(self, spawn_with_specs):
        """A spec table naming an attribute the module lacks exits instead of serving."""
        server = spawn_with_specs(f"{SMOKE_MODULE}.no_such_attr")
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "AttributeError" in logs
        assert "no_such_attr" in logs

    async def test_a_spec_whose_env_raises_fails_fast(self, spawn_with_specs):
        """A spec that cannot compute its env exits instead of serving a worker without it."""
        server = spawn_with_specs(SMOKE_RAISING_SPECS_PATH)
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "RuntimeError" in logs
        assert ENV_VAR_FN_FAILURE_MESSAGE in logs
