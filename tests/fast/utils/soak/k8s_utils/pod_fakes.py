import contextlib
import io
import signal
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from kubernetes_asyncio import client

from tests.utils.soak.k8s_utils import pod_manipulation, pod_processes, pod_processes_cli
from tests.utils.soak.k8s_utils.pod_manipulation import SoakPodTarget

from miles.utils.workers.env_vars import POD_UID_ENV_VAR

_POD_UID = "uid-pod-a"
_BOOT_ID = "boot-a"
_PID_NAMESPACE = "pid:[4026531836]"
_INIT_START_TICKS = 11
_OWN_PID = 2
_PARENT_PID = 3
_FIRST_FD = 1000

# ============================== fake kernel ==============================


@dataclass
class _FakeProcess:
    pid: int
    start_ticks: int
    cmdline: str
    thread_states: dict[int, str]
    exited: bool = False
    exits_on_stop: bool = False
    survives_kill: bool = False
    unstoppable_tids: frozenset[int] = frozenset()
    send_errors: dict[signal.Signals, BaseException] = field(default_factory=dict)


class _FakeProcKernel:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pod_uid = _POD_UID
        self.pid_namespace = _PID_NAMESPACE
        self.processes: dict[int, _FakeProcess] = {}
        self.journal: list[tuple[str, int, str]] = []
        self.open_fds: set[int] = set()
        self.clock = 0.0
        self._pid_of_fd: dict[int, int] = {}
        self._next_fd = _FIRST_FD

        self.set_boot_id(_BOOT_ID)
        self.add(1, cmdline="/sbin/init", start_ticks=_INIT_START_TICKS)

    def add(
        self, pid: int, *, cmdline: str, start_ticks: int = 100, thread_count: int = 2, **overrides: object
    ) -> _FakeProcess:
        process = _FakeProcess(
            pid=pid,
            start_ticks=start_ticks,
            cmdline=cmdline,
            thread_states={pid + offset: "S" for offset in range(thread_count)},
            **overrides,
        )
        self.processes[pid] = process
        self.write(process)
        return process

    def write(self, process: _FakeProcess) -> None:
        directory = self.root / "proc" / str(process.pid)
        (directory / "task").mkdir(parents=True, exist_ok=True)
        (directory / "cmdline").write_bytes(process.cmdline.replace(" ", "\0").encode() + b"\0")
        (directory / "stat").write_text(_stat_line(process, state=process.thread_states[process.pid]))
        for tid, state in process.thread_states.items():
            (directory / "task" / str(tid)).mkdir(exist_ok=True)
            (directory / "task" / str(tid) / "stat").write_text(_stat_line(process, state=state))

    def set_boot_id(self, boot_id: str) -> None:
        path = self.root / "proc" / "sys" / "kernel" / "random" / "boot_id"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{boot_id}\n")

    def restart_init(self, start_ticks: int) -> None:
        self.processes[1].start_ticks = start_ticks
        self.write(self.processes[1])

    def signals(self) -> list[tuple[int, str]]:
        return [(pid, name) for action, pid, name in self.journal if action == "signal"]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pod_processes, "Path", self._path)
        monkeypatch.setattr(
            pod_processes,
            "os",
            SimpleNamespace(
                environ={POD_UID_ENV_VAR: self.pod_uid},
                getpid=lambda: _OWN_PID,
                getppid=lambda: _PARENT_PID,
                readlink=self._readlink,
                pidfd_open=self._pidfd_open,
                close=self._close,
            ),
        )
        monkeypatch.setattr(
            pod_processes,
            "signal",
            SimpleNamespace(
                SIGKILL=signal.SIGKILL,
                SIGSTOP=signal.SIGSTOP,
                SIGCONT=signal.SIGCONT,
                pidfd_send_signal=self._pidfd_send_signal,
            ),
        )
        monkeypatch.setattr(pod_processes, "select", SimpleNamespace(select=self._select))
        monkeypatch.setattr(pod_processes, "time", SimpleNamespace(monotonic=lambda: self.clock, sleep=self._sleep))

    def _path(self, raw: str) -> Path:
        return self.root / str(raw).lstrip("/")

    def _readlink(self, path: str) -> str:
        assert path == "/proc/self/ns/pid"
        return self.pid_namespace

    def _pidfd_open(self, pid: int) -> int:
        if pid not in self.processes:
            raise ProcessLookupError(pid)
        fd = self._next_fd
        self._next_fd += 1
        self._pid_of_fd[fd] = pid
        self.open_fds.add(fd)
        self.journal.append(("open", pid, ""))
        return fd

    def _close(self, fd: int) -> None:
        self.open_fds.remove(fd)
        self.journal.append(("close", self._pid_of_fd[fd], ""))

    def _pidfd_send_signal(self, fd: int, signum: signal.Signals) -> None:
        assert fd in self.open_fds, "Signal sent through a closed pidfd"
        process = self.processes[self._pid_of_fd[fd]]
        if (error := process.send_errors.get(signum)) is not None:
            raise error
        if process.exited:
            raise ProcessLookupError(process.pid)
        self.journal.append(("signal", process.pid, signum.name))

        if signum == signal.SIGKILL and not process.survives_kill:
            process.exited = True
        elif signum == signal.SIGSTOP and process.exits_on_stop:
            process.exited = True
        elif signum == signal.SIGSTOP:
            for tid in process.thread_states:
                if tid not in process.unstoppable_tids:
                    process.thread_states[tid] = "T"
        elif signum == signal.SIGCONT:
            for tid in process.thread_states:
                process.thread_states[tid] = "S"
        self.write(process)

    def _select(
        self, readers: list[int], writers: list[int], errors: list[int], timeout: float
    ) -> tuple[list[int], list[int], list[int]]:
        assert timeout >= 0
        return [fd for fd in readers if self.processes[self._pid_of_fd[fd]].exited], [], []

    def _sleep(self, seconds: float) -> None:
        self.clock += seconds


def _stat_line(process: _FakeProcess, *, state: str) -> str:
    return f"{process.pid} (py (worker) x) {state} " + " ".join(["0"] * 18 + [str(process.start_ticks), "0"]) + "\n"


# ============================== fake CoreV1 ==============================


def _pod_target(**overrides: object) -> SoakPodTarget:
    return SoakPodTarget(**{"namespace": "ns", "release": "rel", "name": "pod-a", "uid": "uid-a", **overrides})


def _live_pod(uid: str, *, resource_version: str | None = "rv-7", deleting: bool = False) -> client.V1Pod:
    return client.V1Pod(
        metadata=client.V1ObjectMeta(
            uid=uid, resource_version=resource_version, deletion_timestamp="2026-09-26T00:00:00Z" if deleting else None
        )
    )


def _api_error(status: int) -> client.ApiException:
    return client.ApiException(status=status, reason="scripted")


class _FakePodApi:
    def __init__(
        self, *, reads: list[client.V1Pod | BaseException], delete_error: BaseException | None = None
    ) -> None:
        self._reads = reads
        self._delete_error = delete_error
        self.calls: list[tuple[str, dict]] = []

    async def read_namespaced_pod(self, *, name: str, namespace: str) -> client.V1Pod:
        self.calls.append(("read", {"name": name, "namespace": namespace}))
        reply = self._reads.pop(0) if len(self._reads) > 1 else self._reads[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def delete_namespaced_pod(self, *, name: str, namespace: str, body: client.V1DeleteOptions) -> None:
        self.calls.append(("delete", {"name": name, "namespace": namespace, "body": body}))
        if self._delete_error is not None:
            raise self._delete_error


def _patch_pod_api(monkeypatch: pytest.MonkeyPatch, api: _FakePodApi) -> None:
    @asynccontextmanager
    async def _core_v1_api() -> AsyncIterator[_FakePodApi]:
        yield api

    monkeypatch.setattr(pod_manipulation, "core_v1_api", _core_v1_api)


# ============================= kubectl exec ==============================


class _FakeKubectlExec:
    def __init__(self, *, replies: list[subprocess.CompletedProcess[str]] | None = None) -> None:
        self._replies = replies
        self.calls: list[dict] = []

    def __call__(
        self,
        argv: list[str],
        *,
        capture_output: bool,
        check: bool,
        input: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(
            {"argv": argv, "capture_output": capture_output, "check": check, "input": input, "timeout": timeout}
        )
        if self._replies is not None:
            return self._replies.pop(0)

        separator = argv.index("--")
        assert argv[separator + 1 : separator + 4] == ["python3", "-m", "tests.utils.soak.k8s_utils.pod_processes_cli"]
        operation, request_id = argv[separator + 4 :]
        stdout, stdin = io.StringIO(), sys.stdin
        sys.stdin = io.StringIO(input)
        try:
            with contextlib.redirect_stdout(stdout):
                {"kill": pod_processes_cli.kill, "stop": pod_processes_cli.stop}[operation](request_id)
        except (AssertionError, OSError) as error:
            return subprocess.CompletedProcess(argv, 1, stdout=stdout.getvalue(), stderr=repr(error))
        finally:
            sys.stdin = stdin
        return subprocess.CompletedProcess(argv, 0, stdout=stdout.getvalue(), stderr="")


def _completed(*, returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["kubectl"], returncode, stdout=stdout, stderr=stderr)
