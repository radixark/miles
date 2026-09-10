"""
Failure modes modeled after torchft's failure.py:
https://github.com/meta-pytorch/torchft/blob/main/examples/monarch/utils/failure.py
"""

import ctypes
import errno
import fcntl
import logging
import os
import signal
import threading
from contextlib import ExitStack
from enum import Enum
from pathlib import Path
from typing import NoReturn

from miles.utils.test_utils.fault_witness import (
    DeadlockTarget,
    witness_deadlock,
    witness_process_exit,
    witness_process_stop,
)

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    SIGKILL = "sigkill"
    EXIT = "exit"
    SEGFAULT = "segfault"
    DEADLOCK = "deadlock"
    THREAD_DEADLOCK = "thread_deadlock"
    SIGSTOP = "sigstop"


def inject_fault(mode: str, *, request_id: str | None = None, receipt_url: str | None = None) -> None:
    failure_mode = FailureMode(mode)
    logger.warning("FaultInjector: executing %s (pid=%d)", failure_mode.value, os.getpid())
    if request_id is not None:
        logger.info("Fault injection request_id=%s mode=%s pid=%d", request_id, mode, os.getpid())

    if failure_mode in {FailureMode.DEADLOCK, FailureMode.THREAD_DEADLOCK}:
        _execute_deadlock(
            holds_gil=failure_mode is FailureMode.DEADLOCK, request_id=request_id, receipt_url=receipt_url
        )

    with ExitStack() as resources:
        if receipt_url is not None:
            if request_id is None:
                raise ValueError("Fault receipts require a request ID")
            witness = witness_process_stop if failure_mode is FailureMode.SIGSTOP else witness_process_exit
            resources.enter_context(witness(request_id=request_id, receipt_url=receipt_url))
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

        case FailureMode.DEADLOCK | FailureMode.THREAD_DEADLOCK:
            _execute_deadlock(holds_gil=failure_mode is FailureMode.DEADLOCK, request_id=None, receipt_url=None)

        case FailureMode.SIGSTOP:
            os.kill(os.getpid(), signal.SIGSTOP)


def _execute_deadlock(*, holds_gil: bool, request_id: str | None, receipt_url: str | None) -> NoReturn:
    if receipt_url is not None and request_id is None:
        raise ValueError("Deadlock receipts require a request ID")
    with ExitStack() as resources:
        held_fd = os.memfd_create("miles-fault-deadlock", os.MFD_CLOEXEC)
        resources.callback(os.close, held_fd)
        waiting_fd = os.open(Path(f"/proc/self/fd/{held_fd}"), os.O_RDWR | os.O_CLOEXEC)
        resources.callback(os.close, waiting_fd)
        fcntl.flock(held_fd, fcntl.LOCK_EX)
        info = os.fstat(held_fd)
        target = DeadlockTarget(
            thread_id=threading.get_native_id(), device=info.st_dev, inode=info.st_ino, holds_gil=holds_gil
        )
        if receipt_url is not None:
            assert request_id is not None
            resources.enter_context(witness_deadlock(request_id=request_id, receipt_url=receipt_url, target=target))
        libc = ctypes.PyDLL(None, use_errno=True) if holds_gil else ctypes.CDLL(None, use_errno=True)
        libc.flock.argtypes = (ctypes.c_int, ctypes.c_int)
        libc.flock.restype = ctypes.c_int
        while True:
            if libc.flock(waiting_fd, fcntl.LOCK_EX) == 0:
                raise RuntimeError("A self-deadlock unexpectedly acquired its lock")
            if (error := ctypes.get_errno()) != errno.EINTR:
                raise OSError(error, os.strerror(error))
