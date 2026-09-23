from typing import Annotated, Union

from pydantic import Discriminator

from miles.utils.test_utils.fault_injector.actions.process import ObserveAction

FaultAction = Annotated[Union[ObserveAction], Discriminator("kind")]
