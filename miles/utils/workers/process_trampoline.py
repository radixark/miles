import ctypes
import os
import signal
import sys

_PR_SET_PDEATHSIG = 1
_EXEC_FAILURE_EXIT_CODE = 127


def main() -> None:
    expected_parent_pid = int(sys.argv[1])
    argv = sys.argv[2:]

    if sys.platform == "linux":
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
        if (parent_pid := os.getppid()) != expected_parent_pid:
            _log(f"parent {expected_parent_pid} is gone (current parent {parent_pid}); exiting without running {argv}")
            os._exit(1)

    _log(f"bound to parent {expected_parent_pid}; exec {argv}")
    try:
        os.execvp(argv[0], argv)
    except OSError as error:
        _log(f"exec of {argv} failed: {error}")
        os._exit(_EXEC_FAILURE_EXIT_CODE)


def _log(message: str) -> None:
    print(f"[process_trampoline pid={os.getpid()}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
