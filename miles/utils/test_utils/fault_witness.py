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
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

logger = logging.getLogger(__name__)

WITNESS_DIRECTORY_ENV = "MILES_FAULT_WITNESS_DIRECTORY"


@contextmanager
def witness_process_exit(*, request_id: str, receipt_url: str) -> Iterator[None]:
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
                json.dumps({"request_id": request_id, "receipt_url": receipt_url, "pid": os.getpid()}).encode()
            )
        if not select.select([ready_read], [], [], 5.0)[0] or os.read(ready_read, 1) != b"1":
            raise TimeoutError("Fault exit witness did not become ready")
        yield
    finally:
        if process is not None:
            _stop_witness(process)
        os.close(pidfd)
        os.close(ready_read)
        if ready_write >= 0:
            os.close(ready_write)


def publish_exit_receipt(*, receipt_url: str, request_id: str, exited_pids: list[int]) -> None:
    request = Request(
        f"{receipt_url.rstrip('/')}/api/v1/fault-receipts/{quote(request_id, safe='')}",
        data=json.dumps({"exited_pids": exited_pids}).encode(),
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
            if not select.select([pidfd], [], [], 10.0)[0]:
                raise TimeoutError("Fault target did not exit after witness initialization")
            publish_exit_receipt(
                receipt_url=config["receipt_url"], request_id=config["request_id"], exited_pids=[config["pid"]]
            )
    finally:
        os.close(pidfd)
        if ready_fd >= 0:
            os.close(ready_fd)


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
