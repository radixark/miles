"""
Failure modes modeled after torchft's failure.py:
https://github.com/meta-pytorch/torchft/blob/main/examples/monarch/utils/failure.py
"""

import ctypes
import logging
import os
import signal
from enum import Enum
from typing import Literal

from miles.utils.test_utils.fault_injector.actions.base import BaseFaultAction, FaultHookContext, FaultHookResources
from miles.utils.workers import process_utils

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    SIGKILL = "sigkill"
    EXIT = "exit"
    SEGFAULT = "segfault"
    DEADLOCK = "deadlock"


def inject_fault(mode: str) -> None:
    failure_mode = FailureMode(mode)
    logger.warning("FaultInjector: executing %s (pid=%d)", failure_mode.value, os.getpid())

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


class ObserveAction(BaseFaultAction):
    kind: Literal["observe"] = "observe"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        return None


class KillProcessAction(BaseFaultAction):
    kind: Literal["kill_process"] = "kill_process"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _signal_process(resources, signal.SIGKILL)


class ExitProcessAction(BaseFaultAction):
    kind: Literal["exit_process"] = "exit_process"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _assert_own_process(resources)
        os._exit(1)


class SegfaultProcessAction(BaseFaultAction):
    kind: Literal["segfault_process"] = "segfault_process"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _assert_own_process(resources)
        ctypes.CFUNCTYPE(None)()()


def _signal_process(resources: FaultHookResources, signum: signal.Signals) -> None:
    if (process := resources.managed_process) is not None:
        process_utils.signal_process_tree(process, signum)
        return
    os.kill(os.getpid(), signum)


def _assert_own_process(resources: FaultHookResources) -> None:
    assert resources.managed_process is None, "This fault is one a process inflicts on itself, not on a subprocess"
