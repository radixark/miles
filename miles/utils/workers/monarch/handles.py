"""How the driver reads the answer of a Monarch endpoint call, the way it reads a Ray actor call's."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

# TODO: import SupervisionError from a public Monarch module once Monarch exports it
from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
from monarch.actor import ActorError

from miles.utils.workers.worker_handle import _WAIT_DEAD_PROBE_INTERVAL_SECONDS, WorkerUnreachableError

if TYPE_CHECKING:
    # runtime.py imports this module, so the worker actor's class is imported for annotations only
    from miles.utils.workers.monarch.runtime import MonarchWorkerActor


async def probe_cell_is_dead(cell_actor_mesh: MonarchWorkerActor) -> bool:
    """Ask the whole cell, which lives and dies as one: a mesh that lost a process fails a call at once, while a call
    to one worker of it can hang once the mesh is stopped."""
    try:
        await asyncio.wait_for(cell_actor_mesh.ping.call(), timeout=_WAIT_DEAD_PROBE_INTERVAL_SECONDS)
    except (TimeoutError, asyncio.TimeoutError):
        return False
    except (SupervisionError, ActorError):
        return True
    return False


async def call_endpoint(future: Any, *, describe: str) -> Any:
    """Answer what the call answered, raising the actor's own exception the way a Ray actor call does."""
    try:
        return await future
    except (ActorError, SupervisionError) as e:
        raise _caller_error(e, describe=describe) from e


def get_endpoint(future: Any, *, describe: str) -> Any:
    """``call_endpoint`` for a caller that blocks on the answer."""
    try:
        return future.get()
    except (ActorError, SupervisionError) as e:
        raise _caller_error(e, describe=describe) from e


def _caller_error(e: ActorError | SupervisionError, *, describe: str) -> Exception:
    """The exception a caller sees for a failed endpoint call.

    - the actor's process died or is unreachable (SupervisionError): WorkerUnreachableError, as for a dead Ray actor;
    - the endpoint raised an Exception: that same exception, e.g. the ValueError the worker raised;
    - the endpoint raised another BaseException, e.g. a CancelledError: a RuntimeError naming it.
    """
    if isinstance(e, SupervisionError):
        return WorkerUnreachableError(f"The actor died or is unreachable when calling {describe!r}: {e!r}")
    if isinstance(e.exception, Exception):
        return e.exception
    # A CancelledError (or other BaseException) raised inside the actor must not read as the caller's own
    return RuntimeError(f"The actor raised {e.exception!r} when calling {describe!r}")
