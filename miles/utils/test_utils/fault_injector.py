"""
Failure modes modeled after torchft's failure.py:
https://github.com/meta-pytorch/torchft/blob/main/examples/monarch/utils/failure.py
"""

import ctypes
import logging
import os
import signal
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    SIGKILL = "sigkill"
    EXIT = "exit"
    SEGFAULT = "segfault"
    DEADLOCK = "deadlock"
    THREAD_DEADLOCK = "thread_deadlock"
    SIGSTOP = "sigstop"


def inject_fault(mode: str) -> None:
    failure_mode = FailureMode(mode)
    logger.warning("FaultInjector: executing %s (pid=%d)", failure_mode.value, os.getpid())

    match failure_mode:
        case FailureMode.THREAD_DEADLOCK:
            threading.Event().wait()

        case FailureMode.SIGSTOP:
            os.kill(os.getpid(), signal.SIGSTOP)

        case FailureMode.SIGKILL:
            os.kill(os.getpid(), signal.SIGKILL)

        case FailureMode.EXIT:
            os._exit(1)

        case FailureMode.SEGFAULT:
            crash_func = ctypes.CFUNCTYPE(None)()
            crash_func()

        case FailureMode.DEADLOCK:
            libc = ctypes.PyDLL(None)
            libc.sleep.argtypes = (ctypes.c_uint,)
            libc.sleep.restype = ctypes.c_uint
            libc.sleep(600)
