from typing import Annotated

from pydantic import Discriminator

from miles.utils.test_utils.fault_injector.actions.process import ObserveAction

FaultAction = Annotated[ObserveAction, Discriminator("kind")]
