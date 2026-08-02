from miles.utils.pydantic_utils import FrozenStrictBaseModel


class StateUnknown(FrozenStrictBaseModel):
    pass


class StatePendingWeights(FrozenStrictBaseModel):
    server_url: str
    bootstrap_port: int | None


class StateServing(FrozenStrictBaseModel):
    server_url: str


# TODO: improve state definitions
CellState = StateUnknown | StatePendingWeights | StateServing
