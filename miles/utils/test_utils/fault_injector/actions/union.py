from typing import Annotated

from pydantic import Discriminator

from miles.utils.test_utils.fault_injector.actions.process import (
    ExitProcessAction,
    KillProcessAction,
    ObserveAction,
    SegfaultProcessAction,
)

FaultAction = Annotated[
    ObserveAction | KillProcessAction | ExitProcessAction | SegfaultProcessAction,
    Discriminator("kind"),
]
