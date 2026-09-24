import ctypes
import os
import signal
import subprocess
import sys

_PR_SET_PDEATHSIG = 1
_EXEC_FAILURE_EXIT_CODE = 127


def main() -> None:
    expected_parent_pid = int(sys.argv[1])
    argv = sys.argv[2:]

    if sys.platform == "linux":
        # Stay as group leader to kill forked descendants if the parent dies.
        # On graceful SIGTERM, let the child use its shutdown grace period.
        signal.signal(signal.SIGTERM, _wait_for_child)
        # Use a separate signal for abrupt parent death, which needs to reap
        # descendants immediately even if the child has forked.
        signal.signal(signal.SIGUSR2, _kill_process_group)
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGUSR2)
        if (parent_pid := os.getppid()) != expected_parent_pid:
            _log(f"parent {expected_parent_pid} is gone (current parent {parent_pid}); exiting without running {argv}")
            os._exit(1)

        _log(f"bound to parent {expected_parent_pid}; supervise {argv}")
        try:
            child = subprocess.Popen(argv)
        except OSError as error:
            _log(f"launch of {argv} failed: {error}")
            os._exit(_EXEC_FAILURE_EXIT_CODE)
        returncode = child.wait()
        if returncode < 0:
            # Keep the signal visible to CommandActor's process.wait(). A
            # plain exit(1) would hide crashes and OOM kills in its logs.
            signum = -returncode
            if signum not in (signal.SIGKILL, signal.SIGSTOP):
                signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            os._exit(1)
        os._exit(returncode if returncode <= 255 else 1)

    _log(f"bound to parent {expected_parent_pid}; exec {argv}")
    try:
        os.execvp(argv[0], argv)
    except OSError as error:
        _log(f"exec of {argv} failed: {error}")
        os._exit(_EXEC_FAILURE_EXIT_CODE)


def _kill_process_group(_signal_number: int, _frame: object) -> None:
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _wait_for_child(_signal_number: int, _frame: object) -> None:
    pass


def _log(message: str) -> None:
    print(f"[process_trampoline pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
