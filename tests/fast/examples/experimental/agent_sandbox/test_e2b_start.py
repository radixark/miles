import asyncio
import enum
import json
import os
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import pytest
from examples.experimental.agent_sandbox import e2b_start
from tests.fast.rollout.agentic.test_worker import _serve_in_thread

from miles.rollout.agentic.agent_function import _CHILD_BOOTSTRAP, InfraAbort, call_agent_function, cancel_calls
from miles.rollout.agentic.worker import create_app
from miles.utils.test_utils import agent_function_stubs as stubs

_DESCENDANTS = r"""
_descendant_cmdlines() {
  local todo=("$ROOT") pid child
  while [ ${#todo[@]} -gt 0 ]; do
    pid=${todo[0]}; todo=("${todo[@]:1}")
    for child in $(cat /proc/$pid/task/*/children 2>/dev/null); do echo /proc/$child/cmdline; todo+=("$child"); done
  done
}
export -f _descendant_cmdlines; export ROOT=$0
"""

FileType = enum.Enum("FileType", {"FILE": "file", "DIR": "dir"})


class FakeSandbox:
    created: list = []
    killed: list = []
    files_on_disk: dict = {}
    worker_host = ""
    write_error: Exception | None = None
    kill_errors: list = []
    on_kill = staticmethod(lambda sandbox: None)

    def __init__(self, kwargs: dict):
        self.sandbox_id = f"sbx-{len(FakeSandbox.created)}"
        self.kwargs, self.written, self.hosts = kwargs, {}, []
        self.files = SimpleNamespace(write=self._write, read=self._read, list=self._list)
        self.commands = SimpleNamespace(run=self._run)

    @classmethod
    def create(cls, **kwargs):
        sandbox = cls(kwargs)
        cls.created.append(sandbox)
        return sandbox

    def get_host(self, port: int) -> str:
        self.hosts.append(port)
        return FakeSandbox.worker_host

    def kill(self):
        if FakeSandbox.kill_errors:
            raise FakeSandbox.kill_errors.pop(0)
        FakeSandbox.on_kill(self)
        FakeSandbox.killed.append(self.sandbox_id)

    def _write(self, path, data):
        if FakeSandbox.write_error is not None:
            raise FakeSandbox.write_error
        self.written[path] = data

    def _read(self, path, format="text"):
        if path not in FakeSandbox.files_on_disk:
            raise FileNotFoundError(path)
        return FakeSandbox.files_on_disk[path]

    def _list(self, path):
        names = {p[len(path) + 1 :].split("/")[0] for p in FakeSandbox.files_on_disk if p.startswith(f"{path}/")}
        return [SimpleNamespace(name=name, type=FileType.DIR) for name in sorted(names)]

    def _run(self, cmd, timeout=60, **kwargs):
        assert e2b_start.AGENT_PROCESS_PROBE in cmd
        scoped = cmd.replace("/proc/[0-9]*/cmdline", "$(_descendant_cmdlines)")
        subprocess.run(["bash", "-c", _DESCENDANTS + scoped, str(os.getpid())], timeout=timeout, check=True)


@pytest.fixture
def service(monkeypatch):
    url, close = _serve_in_thread(create_app())
    monkeypatch.setitem(sys.modules, "e2b", types.SimpleNamespace(Sandbox=FakeSandbox, FileType=FileType))
    monkeypatch.setattr(FakeSandbox, "created", [])
    monkeypatch.setattr(FakeSandbox, "killed", [])
    monkeypatch.setattr(FakeSandbox, "files_on_disk", {})
    monkeypatch.setattr(FakeSandbox, "worker_host", url.removeprefix("http://"))
    monkeypatch.setattr(FakeSandbox, "kill_errors", [])
    monkeypatch.setattr(e2b_start.time, "sleep", lambda s: None)
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    monkeypatch.setenv("SANDBOX_WORKER_TEMPLATE", "agent-worker")
    monkeypatch.setenv("SANDBOX_WORKER_OWNER", "run-7")
    monkeypatch.setenv("SANDBOX_WORKER_URL_SCHEME", "http")
    yield
    close()


START_PATH = "examples.experimental.agent_sandbox.e2b_start.start"


def _in_sandbox(fn, **kwargs):
    return asyncio.run(call_agent_function(fn, mode="worker", worker=START_PATH, **kwargs))


def test_the_call_runs_on_the_worker_in_its_own_sandbox_which_is_then_killed(service):
    assert _in_sandbox(stubs.report_pid)["pid"]

    (sandbox,) = FakeSandbox.created
    assert sandbox.kwargs == {
        "template": "agent-worker",
        "timeout": 7800,
        "metadata": {"owner": "run-7", "role": "agent-worker", "call": "report_pid"},
        "api_key": "test-key",
    }
    assert json.loads(sandbox.written[e2b_start.BOOT_CONFIG]) == {"env": {"E2B_API_KEY": "test-key"}}
    assert sandbox.hosts == [e2b_start.WORKER_PORT]
    assert FakeSandbox.killed == ["sbx-0"]


def test_the_boot_env_file_joins_the_key(service, tmp_path, monkeypatch):
    env_file = tmp_path / "boot.json"
    env_file.write_text(json.dumps({"SESSION_FORWARD": "spec"}))
    monkeypatch.setenv("SANDBOX_WORKER_BOOT_ENV_FILE", str(env_file))

    _in_sandbox(stubs.report_pid)

    (sandbox,) = FakeSandbox.created
    assert json.loads(sandbox.written[e2b_start.BOOT_CONFIG]) == {
        "env": {"E2B_API_KEY": "test-key", "SESSION_FORWARD": "spec"}
    }


def test_the_hosts_worker_secret_joins_the_boot_env(service, tmp_path, monkeypatch):
    secret_file = tmp_path / "worker-secret"
    secret_file.write_text("per-call-secret\n")
    monkeypatch.setenv("MILES_AGENT_WORKER_SECRET_FILE", str(secret_file))

    _in_sandbox(stubs.report_pid)

    (sandbox,) = FakeSandbox.created
    assert json.loads(sandbox.written[e2b_start.BOOT_CONFIG]) == {
        "env": {"E2B_API_KEY": "test-key", "MILES_AGENT_WORKER_SECRET": "per-call-secret"}
    }


def test_a_malformed_boot_env_file_fails_before_any_sandbox(service, tmp_path, monkeypatch):
    env_file = tmp_path / "boot.json"
    env_file.write_text(json.dumps({"N": 1}))
    monkeypatch.setenv("SANDBOX_WORKER_BOOT_ENV_FILE", str(env_file))

    with pytest.raises(InfraAbort, match="JSON object of strings") as raised:
        _in_sandbox(stubs.report_pid)

    assert raised.value.exit_status == "SandboxWorkerUnavailable"
    assert FakeSandbox.created == []


def test_a_sandbox_whose_boot_env_cannot_be_written_is_killed(service, monkeypatch):
    monkeypatch.setattr(FakeSandbox, "write_error", RuntimeError("envd unreachable"))

    with pytest.raises(InfraAbort) as raised:
        _in_sandbox(stubs.report_pid)

    assert raised.value.exit_status == "SandboxWorkerUnavailable"
    assert FakeSandbox.killed == ["sbx-0"]


def test_the_records_are_copied_to_the_same_path_before_the_kill(service, tmp_path, monkeypatch):
    trials = tmp_path / "trials"
    FakeSandbox.files_on_disk.update(
        {f"{trials}/x/result.json": b'{"reward": 1}', f"{trials}/x/agent/timings.jsonl": b"{}\n"}
    )
    monkeypatch.setenv("SANDBOX_WORKER_RECORDS_DIR", str(trials))
    monkeypatch.setenv("SANDBOX_WORKER_RECORDS", "result.json,agent/timings.jsonl,trial.log")
    copied_at_kill: list = []
    monkeypatch.setattr(
        FakeSandbox, "on_kill", staticmethod(lambda sandbox: copied_at_kill.append(sorted(trials.rglob("*.json*"))))
    )

    _in_sandbox(stubs.report_pid)

    assert (trials / "x" / "result.json").read_bytes() == b'{"reward": 1}'
    assert (trials / "x" / "agent" / "timings.jsonl").read_bytes() == b"{}\n"
    assert not (trials / "x" / "trial.log").exists()
    assert copied_at_kill == [[trials / "x" / "agent" / "timings.jsonl", trials / "x" / "result.json"]]


@pytest.mark.parametrize("cancel", ["task", "rollout_abort"])
def test_a_cancelled_agents_cleanup_finishes_before_the_kill(service, tmp_path, monkeypatch, cancel):
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"
    seen_at_kill: list[bool] = []
    monkeypatch.setattr(FakeSandbox, "on_kill", staticmethod(lambda sandbox: seen_at_kill.append(cleaned_up.exists())))

    async def cancel_once_running():
        task = asyncio.create_task(
            call_agent_function(
                stubs.clean_up_slowly,
                mode="worker",
                worker=START_PATH,
                pid_file=str(pid_file),
                cleaned_up=str(cleaned_up),
                cleanup_s=1.0,
            )
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        if cancel == "task":
            task.cancel()
        else:
            assert await cancel_calls() == 1
        with pytest.raises((asyncio.CancelledError, InfraAbort)):
            await task

    asyncio.run(cancel_once_running())

    assert seen_at_kill == [True]


def test_a_failed_kill_is_retried(service, monkeypatch):
    monkeypatch.setattr(FakeSandbox, "kill_errors", [RuntimeError("502 from the gateway")])

    _in_sandbox(stubs.report_pid)

    assert FakeSandbox.killed == ["sbx-0"]


def test_the_agent_process_probe_matches_a_subproc_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", _CHILD_BOOTSTRAP])
    try:
        time.sleep(0.2)
        assert subprocess.run(["bash", "-c", e2b_start.AGENT_PROCESS_PROBE]).returncode == 0
    finally:
        child.kill()
        child.wait()
