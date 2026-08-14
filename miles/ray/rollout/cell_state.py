from miles.utils.pydantic_utils import FrozenStrictBaseModel


class CellAddrInfo(FrozenStrictBaseModel):
    server_url: str
    bootstrap_port: int | None
    gate_url: str


class StateUninitialized(FrozenStrictBaseModel):
    pass


class StateInitializing(FrozenStrictBaseModel):
    addr_info: CellAddrInfo
    start_time: float


class StatePendingWeights(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StateServing(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StateDisposed(FrozenStrictBaseModel):
    pass


CellState = StateUninitialized | StateInitializing | StatePendingWeights | StateServing | StateDisposed
