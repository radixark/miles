import ray

from pydantic import BaseModel, ConfigDict

from miles.utils.indep_dp import IndepDPInfo


class StateBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class StatePending(StateBase):
    pass


class StateAllocatedBase(StateBase):
    actor_handles: list[ray.actor.ActorHandle]


class StateAllocatedUninitialized(StateAllocatedBase):
    pass


class StateAllocatedAlive(StateAllocatedBase):
    indep_dp_info: IndepDPInfo


class StateStopped(StateBase):
    pass


CellState = StatePending | StateAllocatedUninitialized | StateAllocatedAlive | StateStopped
