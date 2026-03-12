from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from miles.utils.ft.utils.graceful_degrade import graceful_degrade

logger = logging.getLogger(__name__)


class KernelLogReader(Protocol):
    def read_new_lines(self) -> list[str]: ...
    def close(self) -> None: ...


class KmsgFileReader:
    """Read kernel messages from /dev/kmsg via non-blocking fd."""

    def __init__(self, *, kmsg_path: Path = Path("/dev/kmsg")) -> None:
        self._fd: int | None = None
        fd = os.open(kmsg_path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            os.lseek(fd, 0, os.SEEK_END)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def read_new_lines(self) -> list[str]:
        if self._fd is None:
            return []

        lines: list[str] = []
        while True:
            try:
                data = os.read(self._fd, 8192)
                if not data:
                    break
                lines.extend(data.decode("utf-8", errors="replace").splitlines())
            except BlockingIOError:
                break

        return lines

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                logger.debug("Failed to close kmsg fd", exc_info=True)
            self._fd = None


class DmesgSubprocessReader:
    """Read kernel messages via dmesg --since subprocess."""

    def __init__(self, since: datetime | None = None) -> None:
        self._last_dmesg_time: datetime = since or datetime.now(timezone.utc)

    @graceful_degrade(default=[], msg="dmesg read failed")
    def read_new_lines(self) -> list[str]:
        since_str = self._last_dmesg_time.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        new_time = datetime.now(timezone.utc)

        result = subprocess.run(
            ["dmesg", "--since", since_str],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self._last_dmesg_time = new_time

        if result.returncode == 0:
            if result.stdout:
                return result.stdout.strip().splitlines()
        else:
            logger.warning(
                "dmesg returned non-zero returncode=%d stderr=%s",
                result.returncode,
                result.stderr[:500] if result.stderr else "",
            )

        return []

    def close(self) -> None:
        pass
