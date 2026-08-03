from miles.utils.pydantic_utils import FrozenStrictBaseModel


class CellAddrInfo(FrozenStrictBaseModel):
    server_url: str
    bootstrap_port: int | None
    gate_url: str


class StateUnknown(FrozenStrictBaseModel):
    pass


class StatePendingWeights(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


class StateServing(FrozenStrictBaseModel):
    addr_info: CellAddrInfo


# TODO: improve state definitions
CellState = StateUnknown | StatePendingWeights | StateServing
