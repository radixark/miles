from pydantic import BaseModel, ConfigDict

from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.workers.worker_handle import BaseWorkerHandle


class StateBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class StateAllocatedBase(StateBase):
    worker_handles: list[BaseWorkerHandle]


class StateAllocatedUninitialized(StateAllocatedBase):
    pass


class StateAllocatedAlive(StateAllocatedBase):
    indep_dp_info: IndepDPInfo


# TODO may remove this state
class StateAllocatedErrored(StateAllocatedBase):
    indep_dp_info: IndepDPInfo | None


CellState = StateAllocatedUninitialized | StateAllocatedAlive | StateAllocatedErrored
