from miles.utils.pydantic_utils import FrozenStrictBaseModel


class CellAddrInfo(FrozenStrictBaseModel):
    server_url: str
    bootstrap_port: int | None
    # None for an engine miles did not launch: there is no gate to release, the server is
    # already up when the run starts.
    gate_url: str | None


class StateUninitialized(FrozenStrictBaseModel):
    pass


class StateInitializing(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StatePendingWeights(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StateServing(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StateDisposed(FrozenStrictBaseModel):
    pass


CellState = StateUninitialized | StateInitializing | StatePendingWeights | StateServing | StateDisposed
