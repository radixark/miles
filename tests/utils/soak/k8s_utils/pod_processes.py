import os
import re
import select
import signal
import sys
import time
from contextlib import ExitStack
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.env_vars import POD_UID_ENV_VAR


class ProcessSignal(StrEnum):
    KILL = "kill"
    STOP = "stop"


class ProcessIdentity(FrozenStrictBaseModel):
    pid: int = Field(gt=1)
    start_ticks: int


class ProcessTarget(FrozenStrictBaseModel):
    pod_uid: str
    boot_id: str
    pid_namespace: str
    init_start_ticks: int
    pattern: str
    processes: list[ProcessIdentity] = Field(min_length=1)


class ProcessSignalReceipt(FrozenStrictBaseModel):
    kind: Literal["process_signal"] = "process_signal"
    request_id: str = Field(min_length=1)
    target: ProcessTarget
    operation: ProcessSignal
    signalled_pids: list[int] = Field(min_length=1)

    def validate_for(self, *, request_id: str, target: ProcessTarget, operation: ProcessSignal) -> None:
        assert self.request_id == request_id, "Process receipt belongs to another request"
        assert self.target == target, "Process receipt belongs to another incarnation"
        assert self.operation == operation, "Process receipt describes another operation"
        assert self.signalled_pids == [process.pid for process in target.processes], "Process receipt is incomplete"


def observe_processes(*, pod_uid: str, pattern: str) -> ProcessTarget:
    assert os.environ[POD_UID_ENV_VAR] == pod_uid, "Pod identity changed"
    matcher = re.compile(pattern)
    processes = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) in {1, os.getpid(), os.getppid()}:
            continue
        try:
            start_ticks = _start_ticks(int(path.name))
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if matcher.search(command) and _start_ticks(int(path.name)) == start_ticks:
                processes.append(ProcessIdentity(pid=int(path.name), start_ticks=start_ticks))
        except FileNotFoundError:
            continue
    _log(f"Observed pids {sorted(process.pid for process in processes)} matching {pattern!r} in pod {pod_uid}")
    return ProcessTarget(
        pod_uid=pod_uid,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
        init_start_ticks=_start_ticks(1),
        pattern=pattern,
        processes=sorted(processes, key=lambda process: process.pid),
    )


def signal_observed_processes(*, target: ProcessTarget, operation: ProcessSignal) -> list[int]:
    _assert_container_unchanged(target)
    with ExitStack() as resources:
        handles = _open_observed_handles(target=target, operation=operation, resources=resources)
        for fd in handles:
            if select.select([fd], [], [], 0)[0]:
                raise ProcessLookupError("An observed process exited before injection")
        for process, fd in zip(target.processes, handles, strict=True):
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            _log(f"Sent SIGKILL to pid {process.pid}")

        pids = [process.pid for process in target.processes]
        _log(f"Confirming pids {pids} exited")
        _confirm_processes_exited(handles)
        _log(f"Confirmed pids {pids} exited")
    return [process.pid for process in target.processes]


def _assert_container_unchanged(target: ProcessTarget) -> None:
    assert os.environ[POD_UID_ENV_VAR] == target.pod_uid, "Pod identity changed"
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == target.boot_id, "Host rebooted"
    assert os.readlink("/proc/self/ns/pid") == target.pid_namespace, "PID namespace changed"
    assert _start_ticks(1) == target.init_start_ticks, "Container restarted"
    _log(f"Container identity unchanged for pod {target.pod_uid}")


def _open_observed_handles(*, target: ProcessTarget, operation: ProcessSignal, resources: ExitStack) -> list[int]:
    matcher = re.compile(target.pattern)
    handles: list[int] = []
    for process in target.processes:
        fd = os.pidfd_open(process.pid)
        resources.callback(os.close, fd)
        assert _start_ticks(process.pid) == process.start_ticks, "Process identity changed"
        command = (Path("/proc") / str(process.pid) / "cmdline").read_bytes().replace(b"\0", b" ")
        assert matcher.search(command.decode(errors="replace")), "Process command changed"
        handles.append(fd)
        _log(f"Opened pidfd for pid {process.pid}")
    return handles


def _confirm_processes_exited(handles: list[int]) -> None:
    pending = set(handles)
    deadline = time.monotonic() + 5.0
    while pending:
        readable, _, _ = select.select(list(pending), [], [], max(0.0, deadline - time.monotonic()))
        if not readable:
            _log(f"Timed out waiting for {len(pending)} signalled processes to exit")
            raise TimeoutError("Signalled processes did not exit within five seconds")
        pending.difference_update(readable)


def _start_ticks(pid: int) -> int:
    return int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])


def _log(message: str) -> None:
    print(f"[pod_processes] {message}", file=sys.stderr, flush=True)
