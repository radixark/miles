"""Limits shared by every process of one run on this node.

Under ``--custom-agent-function-mode subproc`` every agent-function call has a
process of its own, so a module-level lock or semaphore limits nothing. These
are ``flock`` locks on per-run files instead: the kernel releases a slot when
its holder exits, including when the holder is killed.
"""

import asyncio
import fcntl
import os
import random
import re
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import ray

_POLL_S = 0.2


@asynccontextmanager
async def node_semaphore(name: str, limit: int) -> AsyncIterator[None]:
    """Hold one of ``limit`` slots named ``name`` for the duration of the block."""
    if limit < 1:
        raise ValueError(f"node_semaphore {name!r} needs a positive limit, got {limit}")
    fd = await _acquire_any_slot(name, limit)
    try:
        yield
    finally:
        os.close(fd)  # closing the descriptor releases its lock


@contextmanager
def node_lock(name: str) -> Iterator[None]:
    """Hold the lock named ``name``, blocking the calling thread until it is free."""
    fd = _open_lock_file(f"{name}.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


async def _acquire_any_slot(name: str, limit: int) -> int:
    while True:
        # a random order keeps waiters from all contending for slot 0
        for slot in random.sample(range(limit), limit):
            fd = _open_lock_file(f"{name}.{slot}.lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                os.close(fd)
        await asyncio.sleep(_POLL_S * (0.5 + random.random()))


def _open_lock_file(file_name: str) -> int:
    return os.open(_lock_dir() / re.sub(r"[^A-Za-z0-9._-]", "_", file_name), os.O_CREAT | os.O_RDWR, 0o600)


def _lock_dir() -> Path:
    # the processes of one Ray job share its limits; outside Ray a process is its own run
    scope = ray.get_runtime_context().get_job_id() if ray.is_initialized() else f"pid{os.getpid()}"
    lock_dir = Path(tempfile.gettempdir()) / f"miles-node-limits-{scope}"
    lock_dir.mkdir(exist_ok=True)
    return lock_dir
