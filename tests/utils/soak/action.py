import abc
import asyncio
import logging
import os
import shlex
import signal
import subprocess

from tests.utils.soak.state import SoakActionRequest

logger = logging.getLogger(__name__)


class SoakActionForm(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    async def execute(self, request: SoakActionRequest) -> None: ...


async def run_command(
    args: list[str], *, timeout_seconds: float, check: bool = True
) -> subprocess.CompletedProcess[str]:
    logger.info("EXEC: %s", shlex.join(args))
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    communication = asyncio.create_task(process.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout_seconds)
    except (TimeoutError, asyncio.CancelledError):
        await _idempotent_kill_and_reap(process=process, communication=communication)
        raise
    assert process.returncode is not None
    result = subprocess.CompletedProcess(
        args=args,
        returncode=process.returncode,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )
    if check:
        result.check_returncode()
    return result


async def _idempotent_kill_and_reap(
    *,
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes]],
) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    while not communication.done():
        try:
            await asyncio.shield(communication)
        except asyncio.CancelledError:
            continue
    communication.result()
