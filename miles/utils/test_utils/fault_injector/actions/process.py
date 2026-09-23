import ctypes
import os
import signal
import threading
from typing import Literal

from miles.utils.test_utils.fault_injector.actions.base import BaseFaultAction, FaultHookContext, FaultHookResources
from miles.utils.workers import process_utils

FREEZE_SECONDS: int = 600


class ObserveAction(BaseFaultAction):
    kind: Literal["observe"] = "observe"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        return None


class KillProcessAction(BaseFaultAction):
    kind: Literal["kill_process"] = "kill_process"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _signal_process(resources, signal.SIGKILL)


class StopProcessAction(BaseFaultAction):
    kind: Literal["stop_process"] = "stop_process"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _signal_process(resources, signal.SIGSTOP)


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


class FreezeProcessAction(BaseFaultAction):
    kind: Literal["freeze_process"] = "freeze_process"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _assert_own_process(resources)
        libc = ctypes.PyDLL(None)
        libc.sleep.argtypes = (ctypes.c_uint,)
        libc.sleep.restype = ctypes.c_uint
        libc.sleep(FREEZE_SECONDS)


class DeadlockThreadAction(BaseFaultAction):
    kind: Literal["deadlock_thread"] = "deadlock_thread"

    async def __call__(self, *, context: FaultHookContext, resources: FaultHookResources) -> None:
        _assert_own_process(resources)
        threading.Event().wait()


def _signal_process(resources: FaultHookResources, signum: signal.Signals) -> None:
    if (process := resources.managed_process) is not None:
        process_utils.signal_process_tree(process, signum)
        return
    os.kill(os.getpid(), signum)


def _assert_own_process(resources: FaultHookResources) -> None:
    assert resources.managed_process is None, "This fault is one a process inflicts on itself, not on a subprocess"
