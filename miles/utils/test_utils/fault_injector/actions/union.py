from typing import Annotated, Union

from pydantic import Discriminator

from miles.utils.test_utils.fault_injector.actions.cell import StartCellAction, StopCellAction
from miles.utils.test_utils.fault_injector.actions.process import (
    ExitProcessAction,
    KillProcessAction,
    ObserveAction,
    SegfaultProcessAction,
)

FaultAction = Annotated[
    Union[
        ObserveAction,
        KillProcessAction,
        ExitProcessAction,
        SegfaultProcessAction,
        StopCellAction,
        StartCellAction,
    ],
    Discriminator("kind"),
]
