"""Async subprocess helper with timeout and cleanup."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def run_subprocess_with_timeout(
    cmd: list[str],
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> tuple[bytes, bytes, int]:
    """Run *cmd* as a subprocess, returning (stdout, stderr, returncode).

    On timeout the process is killed and ``asyncio.TimeoutError`` is re-raised
    so callers can decide how to handle it (return a failure result, raise, etc.).

    Raises ``OSError`` if the binary cannot be executed at all.
    """
    logger.info("subprocess: running cmd=%s, timeout=%d", cmd[0], timeout_seconds)
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **({"env": env} if env is not None else {}),
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("subprocess: timed out cmd=%s, timeout=%d, pid=%s", cmd[0], timeout_seconds, process.pid)
        try:
            process.kill()
            await process.wait()
        except Exception:
            logger.warning("subprocess: kill_failed cmd=%s", cmd[0], exc_info=True)
        raise

    assert process.returncode is not None
    logger.info("subprocess: completed cmd=%s, returncode=%d", cmd[0], process.returncode)
    return stdout_bytes, stderr_bytes, process.returncode
