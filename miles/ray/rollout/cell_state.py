from miles.utils.pydantic_utils import FrozenStrictBaseModel


class CellAddrInfo(FrozenStrictBaseModel):
    server_url: str
    bootstrap_port: int | None
    gate_url: str | None


class StateUninitialized(FrozenStrictBaseModel):
    pass


class StateInitializing(FrozenStrictBaseModel):
    addr_info: CellAddrInfo
    start_time: float


class StatePendingWeights(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StateServing(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StateEvalPinned(FrozenStrictBaseModel):
    """Carved out of training for one eval: serves the eval router only and receives no weight update."""

    addr_info: CellAddrInfo
    eval_router_url: str


class StateErrored(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StateDisposed(FrozenStrictBaseModel):
    pass


CellState = (
    StateUninitialized
    | StateInitializing
    | StatePendingWeights
    | StateServing
    | StateEvalPinned
    | StateErrored
    | StateDisposed
)
