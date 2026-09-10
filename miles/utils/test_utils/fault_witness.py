import fcntl
import json
import logging
import os
import select
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

logger = logging.getLogger(__name__)

WITNESS_DIRECTORY_ENV = "MILES_FAULT_WITNESS_DIRECTORY"


@dataclass(frozen=True)
class DeadlockTarget:
    thread_id: int
    device: int
    inode: int
    holds_gil: bool


@contextmanager
def witness_process_exit(*, request_id: str, receipt_url: str) -> Iterator[None]:
    with _witness_process(request_id=request_id, receipt_url=receipt_url, effect="exit"):
        yield


@contextmanager
def witness_process_stop(*, request_id: str, receipt_url: str) -> Iterator[None]:
    with _witness_process(request_id=request_id, receipt_url=receipt_url, effect="stop"):
        yield


@contextmanager
def witness_deadlock(*, request_id: str, receipt_url: str, target: DeadlockTarget) -> Iterator[None]:
    with _witness_process(request_id=request_id, receipt_url=receipt_url, effect="deadlock", deadlock=target):
        yield


@contextmanager
def _witness_process(
    *,
    request_id: str,
    receipt_url: str,
    effect: Literal["exit", "stop", "deadlock"],
    deadlock: DeadlockTarget | None = None,
) -> Iterator[None]:
    pidfd = os.pidfd_open(os.getpid())
    ready_read, ready_write = os.pipe()
    process: subprocess.Popen | None = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "miles.utils.test_utils.fault_witness", str(pidfd), str(ready_write)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            pass_fds=(pidfd, ready_write),
            start_new_session=True,
        )
        os.close(ready_write)
        ready_write = -1
        assert process.stdin is not None
        with process.stdin:
            process.stdin.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "receipt_url": receipt_url,
                        "pid": os.getpid(),
                        "effect": effect,
                        "deadlock": asdict(deadlock) if deadlock is not None else None,
                    }
                ).encode()
            )
        if not select.select([ready_read], [], [], 5.0)[0] or os.read(ready_read, 1) != b"1":
            raise TimeoutError("Fault witness did not become ready")
        yield
    finally:
        if process is not None:
            _stop_witness(process)
        os.close(pidfd)
        os.close(ready_read)
        if ready_write >= 0:
            os.close(ready_write)


def publish_exit_receipt(*, receipt_url: str, request_id: str, exited_pids: list[int]) -> None:
    _publish_receipt(receipt_url=receipt_url, request_id=request_id, payload={"exited_pids": exited_pids})


def publish_stop_receipt(*, receipt_url: str, request_id: str, stopped_pids: list[int]) -> None:
    _publish_receipt(receipt_url=receipt_url, request_id=request_id, payload={"stopped_pids": stopped_pids})


def _publish_receipt(*, receipt_url: str, request_id: str, payload: dict[str, object]) -> None:
    request = Request(
        f"{receipt_url.rstrip('/')}/api/v1/fault-receipts/{quote(request_id, safe='')}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = time.monotonic() + 20.0
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            with urlopen(request, timeout=min(5.0, remaining)) as response:
                if response.status != 200:
                    raise ValueError(f"Unexpected fault receipt response: {response.status}")
            return
        except HTTPError as error:
            if error.code < 500:
                raise
            logger.warning("Fault receipt publication failed: %s", request_id, exc_info=True)
        except (URLError, TimeoutError):
            logger.warning("Fault receipt publication failed: %s", request_id, exc_info=True)
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"Fault receipt publication expired: {request_id}")


def _stop_witness(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


def _main() -> None:
    pidfd, ready_fd = map(int, sys.argv[1:])
    config = json.load(sys.stdin)
    try:
        with _witness_lease():
            if select.select([pidfd], [], [], 0)[0]:
                raise ProcessLookupError("Fault target exited before its witness became ready")
            os.write(ready_fd, b"1")
            os.close(ready_fd)
            ready_fd = -1
            if config["effect"] == "deadlock":
                target = DeadlockTarget(**config["deadlock"])
                evidence = _wait_deadlock(pid=config["pid"], pidfd=pidfd, target=target)
                _publish_receipt(
                    receipt_url=config["receipt_url"],
                    request_id=config["request_id"],
                    payload={
                        "blocked_pid": config["pid"],
                        "blocked_tid": target.thread_id,
                        "lock_device": target.device,
                        "lock_inode": target.inode,
                        "holds_gil": target.holds_gil,
                        "lock_evidence": evidence,
                    },
                )
            elif config["effect"] == "stop":
                wait_process_stopped(pid=config["pid"], pidfd=pidfd)
                publish_stop_receipt(
                    receipt_url=config["receipt_url"], request_id=config["request_id"], stopped_pids=[config["pid"]]
                )
            else:
                if not select.select([pidfd], [], [], 10.0)[0]:
                    raise TimeoutError("Fault target did not exit after witness initialization")
                publish_exit_receipt(
                    receipt_url=config["receipt_url"], request_id=config["request_id"], exited_pids=[config["pid"]]
                )
    finally:
        os.close(pidfd)
        if ready_fd >= 0:
            os.close(ready_fd)


def _wait_deadlock(*, pid: int, pidfd: int, target: DeadlockTarget) -> list[str]:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if (
            select.select([pidfd], [], [], 0)[0]
            or not (Path("/proc") / str(pid) / "task" / str(target.thread_id)).exists()
        ):
            raise ProcessLookupError("Deadlock target exited before its lock wait was witnessed")
        if (evidence := deadlock_evidence(Path("/proc/locks").read_text(), pid=pid, target=target)) is not None:
            if select.select([pidfd], [], [], 0)[0]:
                raise ProcessLookupError("Deadlock target exited during lock observation")
            return evidence
        time.sleep(0.01)
    raise TimeoutError("Deadlock target did not wait on its own lock")


def deadlock_evidence(text: str, *, pid: int, target: DeadlockTarget) -> list[str] | None:
    held: list[tuple[str, str]] = []
    waiting: list[tuple[str, str]] = []
    for line in text.splitlines():
        fields = line.split()
        blocked = len(fields) > 1 and fields[1] == "->"
        if blocked:
            fields.pop(1)
        if len(fields) != 8 or fields[1:5] != ["FLOCK", "ADVISORY", "WRITE", str(pid)] or fields[6:] != ["0", "EOF"]:
            continue
        identity = fields[5].split(":")
        if len(identity) != 3:
            continue
        major, minor, inode = int(identity[0], 16), int(identity[1], 16), int(identity[2])
        if (major, minor, inode) != (os.major(target.device), os.minor(target.device), target.inode):
            continue
        (waiting if blocked else held).append((fields[0], line))
    if len(held) == len(waiting) == 1 and held[0][0] == waiting[0][0]:
        return [held[0][1], waiting[0][1]]
    return None


def wait_process_stopped(*, pid: int, pidfd: int, timeout_seconds: float = 10.0) -> None:
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


@contextmanager
def _witness_lease() -> Iterator[None]:
    if (directory := os.environ.get(WITNESS_DIRECTORY_ENV)) is None:
        yield
        return
    with (Path(directory) / uuid4().hex).open("xb") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        yield


if __name__ == "__main__":
    _main()
