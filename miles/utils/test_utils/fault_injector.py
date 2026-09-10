"""
Failure modes modeled after torchft's failure.py:
https://github.com/meta-pytorch/torchft/blob/main/examples/monarch/utils/failure.py
"""

import ctypes
import logging
import os
import signal
from contextlib import ExitStack
from enum import Enum

from miles.utils.test_utils.fault_witness import witness_process_exit

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    SIGKILL = "sigkill"
    EXIT = "exit"
    SEGFAULT = "segfault"
    DEADLOCK = "deadlock"


def inject_fault(mode: str, *, request_id: str | None = None, receipt_url: str | None = None) -> None:
    failure_mode = FailureMode(mode)
    logger.warning("FaultInjector: executing %s (pid=%d)", failure_mode.value, os.getpid())
    if request_id is not None:
        logger.info("Fault injection request_id=%s mode=%s pid=%d", request_id, mode, os.getpid())

    with ExitStack() as resources:
        if receipt_url is not None:
            if request_id is None or failure_mode is FailureMode.DEADLOCK:
                raise ValueError("Exit receipts require a tracked terminating fault")
            resources.enter_context(witness_process_exit(request_id=request_id, receipt_url=receipt_url))
        _execute_fault(failure_mode)


def _execute_fault(failure_mode: FailureMode) -> None:
    match failure_mode:
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
