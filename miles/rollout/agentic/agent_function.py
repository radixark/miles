"""Call a custom agent function on the rollout loop or in a child process of its own.

Every sample's agent function runs concurrently in the rollout process. On one
event loop, a synchronous step in one episode (a large response to parse, a slow
file write) stalls every other episode, long enough under many episodes for their
sandbox SDK requests to time out. ``subproc`` gives each call a fresh child
process, so a slow step, a crash or leaked state stays inside its own episode.
It is a plain subprocess rather than a Ray task because
``--cluster-backend kubernetes`` runs the rollout without Ray.
"""

import asyncio
import contextlib
import ctypes
import logging
import os
import pickle
import signal
import sys
import threading
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, BinaryIO, NoReturn

from miles.rollout.agentic import rollout_limits
from miles.utils.logging_utils import configure_logger_raw

logger = logging.getLogger(__name__)

AGENT_FUNCTION_MODES = ("subproc", "inline")

# A cancelled agent gets this long to run its own cleanup (closing its sandbox) before its process exits.
_CANCEL_GRACE_S = 120.0
# Once the call is over, cleanup the agent left running (a thread closing a sandbox) gets this long.
_LEFTOVER_CLEANUP_S = 600.0

# the child takes the rollout's import path before it imports anything from miles
_CHILD_BOOTSTRAP = (
    "import pickle, sys; sys.path[:] = pickle.load(sys.stdin.buffer); "
    "from miles.rollout.agentic.agent_function import _child_main; _child_main(int(sys.argv[1]))"
)

_PR_SET_PDEATHSIG = 1

# frames on the child's result pipe: a report sent while the call runs, or the call's outcome
_REPORT_FRAME, _OUTCOME_FRAME = b"r", b"o"
_FRAME_HEADER_BYTES = 9  # frame kind + 8-byte payload length


class _ResultPipe:
    """The child's end of the pipe to the caller; agent threads may report concurrently."""

    def __init__(self, file: BinaryIO):
        self._file = file
        self._lock = threading.Lock()

    def send(self, kind: bytes, payload: bytes) -> None:
        with self._lock:
            if self._file.closed:
                return  # the call is over and the caller has stopped reading
            self._file.write(kind + len(payload).to_bytes(8, "big") + payload)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


# set in a ``subproc`` agent process only (see report_to_caller)
_result_pipe: _ResultPipe | None = None


class _RemoteTraceback(Exception):
    """The agent process's traceback, chained under the error it raised, as concurrent.futures does."""

    def __init__(self, tb: str):
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        return self.tb


def report_to_caller(item: Any) -> bool:
    """Send a picklable ``item`` to the process that made this ``subproc`` agent call.

    The caller gets it right away, while the call still runs, through ``on_child_reports``.
    Returns False, sending nothing, when not inside a ``subproc`` agent call."""
    if _result_pipe is None:
        return False
    _result_pipe.send(_REPORT_FRAME, pickle.dumps(item))
    return True


async def call_agent_function(
    fn: Callable[..., Awaitable[Any]],
    *,
    mode: str,
    on_child_reports: Callable[[list[Any]], None] | None = None,
    **kwargs: Any,
) -> Any:
    """Await ``fn(**kwargs)`` on this loop (``inline``) or in a child process of its own (``subproc``).

    Under ``subproc``, ``on_child_reports`` receives each item the agent sends with ``report_to_caller``."""
    if mode == "inline":
        return await fn(**kwargs)
    # pickled before the process starts, so an unpicklable argument fails here
    call = pickle.dumps(sys.path) + pickle.dumps((fn, kwargs, _CANCEL_GRACE_S, _LEFTOVER_CLEANUP_S))
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _CHILD_BOOTSTRAP,
        str(os.getpid()),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, rollout_limits.SCOPE_ENV_VAR: rollout_limits.current_scope()},
    )
    try:
        await _send_call(proc, call)
        # not cancelled with the call, so reports from the agent's cancel cleanup (its teardown) still arrive
        output = await asyncio.shield(_read_results(proc, on_child_reports))
    except asyncio.CancelledError:
        _terminate(proc)
        raise
    if output is None:
        returncode = await proc.wait()
        raise RuntimeError(f"The agent process exited with code {returncode} before returning a result")
    return _unpack_outcome(output)


async def _send_call(proc: asyncio.subprocess.Process, call: bytes) -> None:
    try:
        proc.stdin.write(call)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass  # the agent process died before reading its call; its exit code says why
    finally:
        proc.stdin.close()


async def _read_results(
    proc: asyncio.subprocess.Process, on_child_reports: Callable[[list[Any]], None] | None
) -> bytes | None:
    """Pass on reports as they arrive and return the pickled outcome, or None if none came.

    The outcome arrives before the process finishes its leftover cleanup."""
    while True:
        try:
            header = await proc.stdout.readexactly(_FRAME_HEADER_BYTES)
            payload = await proc.stdout.readexactly(int.from_bytes(header[1:], "big"))
        except asyncio.IncompleteReadError:
            return None
        if header[:1] == _OUTCOME_FRAME:
            return payload
        if on_child_reports is not None:
            try:
                on_child_reports([pickle.loads(payload)])
            except Exception:
                logger.exception("handling a report from the agent process failed")


def _terminate(proc: asyncio.subprocess.Process) -> None:
    _send_signal(proc, signal.SIGTERM)  # the agent process cancels the agent and enforces both deadlines itself
    # a backstop for an agent process that cannot run Python to enforce them, such as one stuck holding the GIL
    asyncio.get_running_loop().call_later(_CANCEL_GRACE_S + _LEFTOVER_CLEANUP_S, _send_signal, proc, signal.SIGKILL)


def _send_signal(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)


def _unpack_outcome(output: bytes) -> Any:
    outcome = pickle.loads(output)
    if outcome[0] == "ok":
        return outcome[1]
    _, error, child_traceback = outcome
    raise error from _RemoteTraceback(f'\n"""\n{child_traceback}"""')


def _child_main(parent_pid: int) -> None:
    global _result_pipe
    _die_with_parent(parent_pid)
    # Reports and the outcome go to the original stdout; what the agent prints goes to stderr instead.
    _result_pipe = result_pipe = _ResultPipe(os.fdopen(os.dup(1), "wb"))
    os.dup2(2, 1)
    configure_logger_raw(f"agent_function pid={os.getpid()}")
    try:
        fn, kwargs, cancel_grace_s, leftover_cleanup_s = pickle.load(sys.stdin.buffer)
    except Exception as error:
        _send_outcome(result_pipe, ("error", error, traceback.format_exc()))
        _exit(0)
    threads_before = set(threading.enumerate())
    try:
        asyncio.run(_run_and_report(fn, kwargs, result_pipe, cancel_grace_s, leftover_cleanup_s))
    except BaseException:
        traceback.print_exc()
        _exit(1)
    # the deadline armed once the call was over bounds this wait
    for thread in set(threading.enumerate()) - threads_before:
        if not thread.daemon:
            thread.join()
    _exit(0)


def _die_with_parent(parent_pid: int) -> None:
    """Have the kernel kill this process when the rollout process dies, so no agent outlives its rollout."""
    if not sys.platform.startswith("linux"):
        return
    if ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        os._exit(1)  # the rollout process died before the signal was armed


async def _run_and_report(
    fn: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
    result_pipe: _ResultPipe,
    cancel_grace_s: float,
    leftover_cleanup_s: float,
) -> None:
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    cancel_deadlines: list[threading.Timer] = []

    def cancel_on_sigterm() -> None:
        # the caller sends SIGTERM when it is cancelled: cancelling the agent runs its cleanup
        if not cancel_deadlines:
            task.cancel()
            cancel_deadlines.append(_exit_after(cancel_grace_s))

    loop.add_signal_handler(signal.SIGTERM, cancel_on_sigterm)
    try:
        outcome = ("ok", await fn(**kwargs))
    except asyncio.CancelledError:
        outcome = None  # the caller was cancelled and reads no outcome
    except Exception as error:
        outcome = ("error", error, traceback.format_exc())
    except BaseException as error:
        # an agent that exits or is interrupted ends only its own episode, not the rollout
        outcome = ("error", RuntimeError(f"The agent function raised {error!r}"), traceback.format_exc())
    for deadline in cancel_deadlines:
        deadline.cancel()
    # once the call is over, its leftover cleanup runs to its own deadline whatever the caller does
    loop.remove_signal_handler(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # sent before the loop shuts down, which waits for the agent's leftover executor threads
    _send_outcome(result_pipe, outcome)
    _exit_after(leftover_cleanup_s)


def _exit_after(seconds: float) -> threading.Timer:
    timer = threading.Timer(seconds, _exit, args=(1,))
    timer.daemon = True
    timer.start()
    return timer


def _send_outcome(result_pipe: _ResultPipe, outcome: tuple | None) -> None:
    if outcome is not None:
        result_pipe.send(_OUTCOME_FRAME, _dump_outcome(outcome))
    result_pipe.close()  # the caller stops waiting here


def _exit(code: int) -> NoReturn:
    sys.stdout.flush()
    sys.stderr.flush()
    # skips interpreter shutdown, which would wait without limit for a thread still running
    os._exit(code)


def _dump_outcome(outcome: tuple) -> bytes:
    """Pickle the outcome, replacing what the caller could not rebuild."""
    if outcome[0] == "ok":
        try:
            return pickle.dumps(outcome)
        except Exception as e:
            error = TypeError(f"The agent function returned a value that cannot be pickled: {e}")
            outcome = ("error", error, traceback.format_exc())
    _, error, child_traceback = outcome
    try:
        data = pickle.dumps(outcome)
        # some errors pickle but cannot be rebuilt, such as one whose constructor takes keyword-only arguments
        pickle.loads(data)
        return data
    except Exception:
        return pickle.dumps(("error", RuntimeError(f"{type(error).__qualname__}: {error}"), child_traceback))
