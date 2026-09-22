import os
import re
import select
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

import typer
from pydantic import Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.env_vars import POD_UID_ENV_VAR


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
    operation: Literal["kill", "stop"]
    signalled_pids: list[int] = Field(min_length=1)

    def validate_for(self, *, request_id: str, target: ProcessTarget, operation: Literal["kill", "stop"]) -> None:
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
    return ProcessTarget(
        pod_uid=pod_uid,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
        init_start_ticks=_start_ticks(1),
        pattern=pattern,
        processes=sorted(processes, key=lambda process: process.pid),
    )


def _signal_observed_processes(*, target: ProcessTarget, operation: Literal["kill", "stop"]) -> list[int]:
    _assert_container_unchanged(target)
    with ExitStack() as resources, ExitStack() as rollback:
        handles = _open_observed_handles(target=target, operation=operation, resources=resources)
        for fd in handles:
            if select.select([fd], [], [], 0)[0]:
                raise ProcessLookupError("An observed process exited before injection")
        for fd in handles:
            signal.pidfd_send_signal(fd, signal.SIGKILL if operation == "kill" else signal.SIGSTOP)
            if operation == "stop":
                rollback.callback(_idempotent_resume, fd)

        if operation == "stop":
            _confirm_processes_stopped(target=target, handles=handles)
            rollback.pop_all()
            return [process.pid for process in target.processes]
        _confirm_processes_exited(handles)
    return [process.pid for process in target.processes]


def _assert_container_unchanged(target: ProcessTarget) -> None:
    assert os.environ[POD_UID_ENV_VAR] == target.pod_uid, "Pod identity changed"
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == target.boot_id, "Host rebooted"
    assert os.readlink("/proc/self/ns/pid") == target.pid_namespace, "PID namespace changed"
    assert _start_ticks(1) == target.init_start_ticks, "Container restarted"


def _open_observed_handles(
    *, target: ProcessTarget, operation: Literal["kill", "stop"], resources: ExitStack
) -> list[int]:
    matcher = re.compile(target.pattern)
    handles: list[int] = []
    for process in target.processes:
        fd = os.pidfd_open(process.pid)
        resources.callback(os.close, fd)
        assert _start_ticks(process.pid) == process.start_ticks, "Process identity changed"
        command = (Path("/proc") / str(process.pid) / "cmdline").read_bytes().replace(b"\0", b" ")
        assert matcher.search(command.decode(errors="replace")), "Process command changed"
        if operation == "stop" and (
            (Path("/proc") / str(process.pid) / "stat").read_text().rsplit(")", 1)[1].split()[0] == "T"
        ):
            raise ProcessLookupError("An observed process was already stopped")
        handles.append(fd)
    return handles


def _confirm_processes_stopped(*, target: ProcessTarget, handles: list[int]) -> None:
    deadline = time.monotonic() + 5.0
    for process, fd in zip(target.processes, handles, strict=True):
        _wait_process_stopped(pid=process.pid, pidfd=fd, timeout_seconds=deadline - time.monotonic())
    if any(select.select([fd], [], [], 0)[0] for fd in handles):
        raise ProcessLookupError("An observed process exited during stop confirmation")


def _confirm_processes_exited(handles: list[int]) -> None:
    pending = set(handles)
    deadline = time.monotonic() + 5.0
    while pending:
        readable, _, _ = select.select(list(pending), [], [], max(0.0, deadline - time.monotonic()))
        if not readable:
            raise TimeoutError("Signalled processes did not exit within five seconds")
        pending.difference_update(readable)


def _idempotent_resume(fd: int) -> None:
    try:
        signal.pidfd_send_signal(fd, signal.SIGCONT)
    except ProcessLookupError:
        pass


def _wait_process_stopped(*, pid: int, pidfd: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if select.select([pidfd], [], [], 0)[0]:
            raise ProcessLookupError("Fault target exited before stop was witnessed")
        tasks = list((Path("/proc") / str(pid) / "task").glob("*/stat"))
        states = [path.read_text().rsplit(")", 1)[1].split()[0] for path in tasks]
        if states and all(state == "T" for state in states):
            if select.select([pidfd], [], [], 0)[0]:
                raise ProcessLookupError("Fault target exited during stop observation")
            return
        time.sleep(0.01)
    raise TimeoutError("Fault target did not stop after witness initialization")


def _start_ticks(pid: int) -> int:
    return int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])


app = typer.Typer()


@app.command()
def observe(pod_uid: str, pattern: str) -> None:
    print(observe_processes(pod_uid=pod_uid, pattern=pattern).model_dump_json(), flush=True)


@app.command()
def kill(request_id: str) -> None:
    _print_receipt(request_id=request_id, operation="kill")


@app.command()
def stop(request_id: str) -> None:
    _print_receipt(request_id=request_id, operation="stop")


def _print_receipt(*, request_id: str, operation: Literal["kill", "stop"]) -> None:
    target = ProcessTarget.model_validate_json(sys.stdin.read())
    receipt = ProcessSignalReceipt(
        request_id=request_id,
        target=target,
        operation=operation,
        signalled_pids=_signal_observed_processes(target=target, operation=operation),
    )
    print(receipt.model_dump_json(), flush=True)


if __name__ == "__main__":
    app()
