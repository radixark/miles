import os
import signal
from types import SimpleNamespace

import pytest
from tests.fast.utils.test_utils.fault_injector.fakes import _Effects

from miles.utils.test_utils.fault_injector.actions.base import BaseFaultAction, FaultHookContext, FaultHookResources
from miles.utils.test_utils.fault_injector.actions.process import (
    ExitProcessAction,
    KillProcessAction,
    ObserveAction,
    SegfaultProcessAction,
)

_MANAGED = SimpleNamespace(pid=4321)


async def _run(action: BaseFaultAction, *, managed: bool) -> None:
    await action(
        context=FaultHookContext(rollout_id=3),
        resources=FaultHookResources(managed_process=_MANAGED if managed else None),
    )


class TestSignalActions:
    async def test_a_managed_process_tree_receives_the_signal(self, effects: _Effects) -> None:
        """A supervised worker must be signalled as a whole tree while its supervisor survives."""
        await _run(KillProcessAction(), managed=True)
        assert effects.log == [("tree", (4321, signal.SIGKILL))]

    async def test_an_unmanaged_process_signals_itself(self, effects: _Effects) -> None:
        """Without a managed subprocess the signal must hit exactly the current process."""
        await _run(KillProcessAction(), managed=False)
        assert effects.log == [("kill", (os.getpid(), signal.SIGKILL))]

    async def test_observe_has_no_side_effect(self, effects: _Effects) -> None:
        """An observe action must only record that the hook was reached."""
        await _run(ObserveAction(), managed=True)
        await _run(ObserveAction(), managed=False)
        assert effects.log == []


class TestSelfInflictedActions:
    async def test_the_own_process_inflicts_the_fault(self, effects: _Effects) -> None:
        """Exit must act on the calling process without sending any signal."""
        await _run(ExitProcessAction(), managed=False)
        assert effects.log == [("exit", 1)]

    async def test_a_segfault_calls_a_null_function_pointer(self, effects: _Effects) -> None:
        """The segfault must come from calling a void null function, not from a signal."""
        await _run(SegfaultProcessAction(), managed=False)
        assert effects.log == [("segfault", None)]

    @pytest.mark.parametrize("action", [ExitProcessAction(), SegfaultProcessAction()])
    async def test_a_managed_subprocess_cannot_receive_a_self_inflicted_fault(
        self, effects: _Effects, action: BaseFaultAction
    ) -> None:
        """Self-inflicted faults must refuse to run on behalf of a subprocess and touch nothing."""
        with pytest.raises(AssertionError, match="not on a subprocess"):
            await _run(action, managed=True)
        assert effects.log == []
