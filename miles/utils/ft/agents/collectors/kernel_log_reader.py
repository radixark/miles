from __future__ import annotations

import logging
import os
import subprocess
import threading
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
        self._closing = threading.Event()
        logger.info("collector: opening kmsg file reader: path=%s", kmsg_path)
        fd = os.open(kmsg_path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            os.lseek(fd, 0, os.SEEK_END)
        except BaseException:
            os.close(fd)
            logger.error("collector: failed to seek to end of kmsg: path=%s", kmsg_path, exc_info=True)
            raise
        self._fd = fd

    def read_new_lines(self) -> list[str]:
        if self._fd is None or self._closing.is_set():
            logger.debug("collector: kmsg read skipped, fd=%s closing=%s", self._fd is not None, self._closing.is_set())
            return []

        lines: list[str] = []
        while True:
            if self._closing.is_set():
                break
            try:
                data = os.read(self._fd, 8192)
                if not data:
                    break
                lines.extend(data.decode("utf-8", errors="replace").splitlines())
            except OSError:
                break

        if lines:
            logger.debug("collector: kmsg read new lines: count=%d", len(lines))
        return lines

    def close(self) -> None:
        logger.info("collector: closing kmsg file reader")
        self._closing.set()
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                logger.debug("collector: failed to close kmsg fd", exc_info=True)
            self._fd = None


class DmesgSubprocessReader:
    """Read kernel messages via dmesg --since subprocess."""

    def __init__(self, since: datetime | None = None) -> None:
        self._last_dmesg_time: datetime = since or datetime.now(timezone.utc)
        logger.info("collector: dmesg subprocess reader initialized: since=%s", self._last_dmesg_time)

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

        if result.returncode != 0:
            # Do not advance _last_dmesg_time on failure — the next successful
            # call will re-read this window so no kernel logs are permanently lost.
            logger.warning(
                "collector: dmesg returned non-zero: returncode=%d, stderr=%s",
                result.returncode,
                result.stderr[:500] if result.stderr else "",
            )
            return []

        self._last_dmesg_time = new_time

        if result.stdout:
            lines = result.stdout.strip().splitlines()
            logger.debug("collector: dmesg read new lines: count=%d", len(lines))
            return lines

        logger.debug("collector: dmesg returned no output")
        return []

    def close(self) -> None:
        pass
