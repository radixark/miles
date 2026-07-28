import ray
from pydantic import ConfigDict

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class AddrInfo(FrozenStrictBaseModel):
    server_url: str
    bootstrap_port: int | None = None


# ------------------------- states -----------------------------


class StateBase(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class StateStopped(StateBase):
    pass


class StateAllocatedBase(StateBase):
    actor_handle: ray.actor.ActorHandle
    addr_info: AddrInfo | None = None


class StateAllocatedUninitialized(StateAllocatedBase):
    pass


class StateAllocatedAlive(StateAllocatedBase):
    pass


CellState = StateStopped | StateAllocatedUninitialized | StateAllocatedAlive
