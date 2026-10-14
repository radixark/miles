import os
import signal

from tests.fast.utils.test_utils.fault_injector.fakes import _Effects

from miles.utils.test_utils.fault_injector.actions.base import BaseFaultAction, FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.actions.process import (
    ExitProcessAction,
    KillProcessAction,
    ObserveAction,
    SegfaultProcessAction,
)


async def _run(action: BaseFaultAction) -> None:
    await action(context=FaultHookContext(rollout_id=3), resources=FaultHookResources())


class TestSignalActions:
    async def test_the_kill_signals_the_own_process(self, effects: _Effects) -> None:
        """A kill must hit exactly the process that reached the hook."""
        await _run(KillProcessAction())
        assert effects.log == [("kill", (os.getpid(), signal.SIGKILL))]

    async def test_observe_has_no_side_effect(self, effects: _Effects) -> None:
        """An observe action must only record that the hook was reached."""
        await _run(ObserveAction())
        assert effects.log == []


class TestSelfInflictedActions:
    async def test_the_own_process_exits(self, effects: _Effects) -> None:
        """Exit must act on the calling process without sending any signal."""
        await _run(ExitProcessAction())
        assert effects.log == [("exit", 1)]

    async def test_a_segfault_calls_a_null_function_pointer(self, effects: _Effects) -> None:
        """The segfault must come from calling a void null function, not from a signal."""
        await _run(SegfaultProcessAction())
        assert effects.log == [("segfault", None)]
