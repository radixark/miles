import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from tests.fast.utils.workers.e2e.e2e_worker import (
    ENV_VAR_FAILURE_MESSAGE,
    RPC_PORT_FLAG,
    E2eServeSpec,
    E2eWorkerConfig,
    FailingEnvE2eServeSpec,
)
from tests.fast.utils.workers.e2e.harness import (
    READY_TIMEOUT_SECONDS,
    REPO_ROOT,
    ServerProcess,
    port_is_refused,
    reserve_port,
    spawn_serve_process,
)
from tests.fast.utils.workers.serving.registered_serve import pod_env, serve_config_argv

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle


@pytest.fixture
def spawn_with_config(state_dir: Path, tmp_path: Path) -> Iterator[Callable[..., ServerProcess]]:
    started: list[ServerProcess] = []

    def start(*, edit_payload: Callable[[dict], None]) -> ServerProcess:
        port = reserve_port()
        config = E2eWorkerConfig(worker_argv=["--state-dir", str(state_dir), RPC_PORT_FLAG, str(port)])
        (flag, payload) = serve_config_argv(spec_class=E2eServeSpec, config=config)
        edited = json.loads(payload)
        edit_payload(edited)

        server = spawn_serve_process(
            own_argv=[flag, json.dumps(edited)],
            pod_env_vars=pod_env(E2eServeSpec.create(config)),
            port=port,
            log_path=tmp_path / f"config-server-{len(started)}.log",
        )
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

    async def test_the_pool_config_reaches_the_worker(self, handle):
        """The worker is built from the config the launcher serialized for its pool."""
        argv = await handle.report_argv()
        assert "--state-dir" in argv

    async def test_the_pool_config_arrives_verbatim(self, spawn, make_handle):
        """Values that look like separators or flags must reach the worker unchanged."""
        server = spawn(worker_argv=["--flag", "--", "--inner"])
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        argv = await handle.report_argv()
        assert argv[-3:] == ["--flag", "--", "--inner"]

    async def test_the_spec_computes_its_env_from_the_pool_config(self, handle):
        """The spec is rebuilt from the pool's own config, not from the entrypoint's."""
        recorded = await handle.report_env(name="MILES_E2E_ARGV")
        assert "--state-dir" in recorded

    async def test_the_spec_env_is_applied_before_the_inner_worker_is_imported(
        self, spawn: Callable[..., ServerProcess], make_handle: Callable[..., RpcWorkerHandle]
    ) -> None:
        """The worker's module must observe the spec's environment when the exec'd interpreter imports it."""
        server = spawn(extra_env={"MILES_E2E_ARGV": "inherited-before-spec-env"})
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        assert await handle.report_argv_env_at_import() == ",".join(await handle.report_argv())

    async def test_parent_environment_is_inherited(self, spawn, make_handle):
        """Environment from the launcher reaches the worker."""
        server = spawn(extra_env={"MILES_E2E_MARKER": "inherited"})
        handle = make_handle(server)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        assert await handle.report_env(name="MILES_E2E_MARKER") == "inherited"


class TestStartupFailures:
    async def test_an_unknown_worker_type_fails_fast(self, spawn_with_config):
        """A pool whose worker type the image does not know exits instead of serving."""
        server = spawn_with_config(edit_payload=lambda payload: payload.update(worker_type="no-such-worker"))
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "KeyError" in logs
        assert "no-such-worker" in logs

    async def test_missing_config_argument_is_a_usage_error(self):
        """argparse rejects a missing --config with its usage exit code."""
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [sys.executable, "-m", "miles.utils.workers.serving.serve"],
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

    async def test_a_config_missing_a_field_fails_fast(self, spawn_with_config):
        """A pool config the worker type cannot validate exits rather than serving a broken worker."""
        server = spawn_with_config(edit_payload=lambda payload: payload["args"].clear())
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)
        assert "worker_argv" in server.logs()

    async def test_a_config_with_an_unknown_field_fails_fast(self, spawn_with_config):
        """A field the worker type does not declare would be dropped, so the pod exits instead."""
        server = spawn_with_config(edit_payload=lambda payload: payload["args"].update(no_such_field=1))
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "ValidationError" in logs
        assert "no_such_field" in logs

    async def test_a_spec_whose_env_raises_fails_fast(self, spawn):
        """A spec that cannot compute its env exits instead of serving a worker without it."""
        server = spawn(spec_class=FailingEnvE2eServeSpec, wait=False)
        assert server.wait(timeout=30.0) not in (None, 0)
        assert port_is_refused(server.port)

        logs = server.logs()
        assert "RuntimeError" in logs
        assert ENV_VAR_FAILURE_MESSAGE in logs
