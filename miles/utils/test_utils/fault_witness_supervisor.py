import fcntl
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import FrameType

from miles.utils.test_utils.fault_witness import WITNESS_DIRECTORY_ENV

logger = logging.getLogger(__name__)


def supervise(command: list[str]) -> int:
    if not command:
        raise ValueError("A supervised worker command is required")
    with tempfile.TemporaryDirectory(prefix="miles-fault-witness-") as directory:
        process = subprocess.Popen(
            command, env={**os.environ, WITNESS_DIRECTORY_ENV: directory}, start_new_session=True
        )
        shutdown_deadline: float | None = None

        def forward_signal(signum: int, frame: FrameType | None) -> None:
            nonlocal shutdown_deadline
            if shutdown_deadline is None:
                shutdown_deadline = time.monotonic() + 20.0
            if process.poll() is None:
                process.send_signal(signum)

        previous = {sig: signal.signal(sig, forward_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while process.poll() is None:
                if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                    process.kill()
                    process.wait(timeout=5.0)
                    break
                time.sleep(0.1)
            _drain_witnesses(Path(directory))
            assert process.returncode is not None
            return process.returncode if process.returncode >= 0 else 128 - process.returncode
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)


def _drain_witnesses(directory: Path) -> None:
    deadline = time.monotonic() + 25.0
    while True:
        pending = False
        for path in directory.iterdir():
            with path.open("rb") as lease:
                try:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pending = True
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Fault witnesses did not finish before the container exit deadline")
        time.sleep(0.1)


if __name__ == "__main__":
    sys.exit(supervise(sys.argv[1:]))
