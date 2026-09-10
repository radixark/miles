import abc
import asyncio
import logging
import os
import shlex
import signal
import subprocess
from contextlib import ExitStack
from pathlib import Path

from tests.utils.soak.state import SoakActionRequest

logger = logging.getLogger(__name__)


class SoakActionForm(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    async def execute(self, request: SoakActionRequest) -> None: ...


async def run_command(
    args: list[str],
    *,
    timeout_seconds: float,
    check: bool = True,
    stdin_data: str | None = None,
    output_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    logger.info("EXEC: %s", shlex.join(args))
    with ExitStack() as resources:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = resources.enter_context(output_path.open("xb"))
        else:
            output = asyncio.subprocess.PIPE
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
            stdout=output,
            stderr=asyncio.subprocess.STDOUT if output_path is not None else asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communication = asyncio.create_task(
            process.communicate(stdin_data.encode()) if stdin_data is not None else process.communicate()
        )
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout_seconds)
        except (TimeoutError, asyncio.CancelledError):
            await _idempotent_kill_and_reap(process=process, communication=communication)
            raise
    assert process.returncode is not None
    result = subprocess.CompletedProcess(
        args=args,
        returncode=process.returncode,
        stdout=(stdout or b"").decode(errors="replace"),
        stderr=(stderr or b"").decode(errors="replace"),
    )
    if check:
        result.check_returncode()
    return result


async def _idempotent_kill_and_reap(
    *,
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]],
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
