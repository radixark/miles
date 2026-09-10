import abc
import asyncio
import logging
import os
import shlex
import signal
import subprocess
from contextlib import ExitStack
from pathlib import Path
from typing import TypeVar

from tests.utils.soak.state import SoakActionRequest

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


class SoakActionError(RuntimeError):
    def __init__(self, message: str, *, evidence: dict) -> None:
        super().__init__(message)
        self.evidence = evidence


class SoakActionForm(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    async def execute(self, request: SoakActionRequest) -> dict | None: ...


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
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
                stdout=output,
                stderr=asyncio.subprocess.STDOUT if output_path is not None else asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        )
        try:
            process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            try:
                process = await _wait_for_cleanup(spawning)
            except TimeoutError:
                spawning.add_done_callback(_idempotent_close_late_spawn)
                spawning.cancel()
                raise
            await _idempotent_kill_and_reap(process=process, communication=asyncio.create_task(process.communicate()))
            raise
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
    try:
        await _wait_for_cleanup(communication)
    except TimeoutError:
        communication.cancel()
        process._transport.close()
        try:
            await _wait_for_cleanup(communication)
        except asyncio.CancelledError:
            pass
        await _wait_for_cleanup(asyncio.create_task(process.wait()))
        raise


async def _wait_for_cleanup(task: asyncio.Task[_T]) -> _T:
    deadline = asyncio.get_running_loop().time() + 10.0
    while not task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Subprocess cleanup exceeded 10 seconds")
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            continue
    return task.result()


def _idempotent_close_late_spawn(task: asyncio.Task[asyncio.subprocess.Process]) -> None:
    if task.cancelled():
        return
    try:
        process = task.result()
    except Exception:
        logger.warning("Cancelled subprocess creation failed", exc_info=True)
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process._transport.close()
