from typing import Annotated

from pydantic import Discriminator

from miles.utils.test_utils.fault_injector.actions.cell import StartCellAction, StopCellAction
from miles.utils.test_utils.fault_injector.actions.frozen import SleepForeverAction
from miles.utils.test_utils.fault_injector.actions.process import (
    DeadlockThreadAction,
    ExitProcessAction,
    FreezeProcessAction,
    KillProcessAction,
    ObserveAction,
    SegfaultProcessAction,
    StopProcessAction,
)

FaultAction = Annotated[
    ObserveAction
    | KillProcessAction
    | StopProcessAction
    | ExitProcessAction
    | SegfaultProcessAction
    | FreezeProcessAction
    | DeadlockThreadAction
    | StopCellAction
    | StartCellAction
    | SleepForeverAction,
    Discriminator("kind"),
]
