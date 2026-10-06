import asyncio
from typing import TYPE_CHECKING

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from miles.utils.pydantic_utils import FrozenStrictBaseModel, StrictBaseModel

if TYPE_CHECKING:
    from miles.rollout.generate_utils.output_store import ReplayOutputs


class CreateSessionRequest(StrictBaseModel):
    evaluation: StrictBool = False
    # the caller resolves rollout/eval/dataset values; the session only fills fields a request omits
    temperature: StrictFloat | None = None
    top_p: StrictFloat | None = None
    top_k: StrictInt | None = None

    @field_validator("top_k", mode="before")
    @classmethod
    def _integral_float_as_int(cls, value):
        # an eval dataset YAML may spell top_k as 40.0; a fractional top_k stays an error
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value


class SessionServerInstance(FrozenStrictBaseModel):
    """One session-server instance as the driver published it."""

    # ``host:port`` the driver dials.
    addr: str
    # ``host:port`` a peer outside the cluster dials; defaults to ``addr`` (filled in below).
    external_addr: str = ""
    instance_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_external_addr_to_addr(cls, values: dict) -> dict:
        if isinstance(values, dict) and not values.get("external_addr"):
            values = {**values, "external_addr": values.get("addr")}
        return values

    @property
    def url(self) -> str:
        return f"http://{self.addr}"

    @property
    def external_url(self) -> str:
        return f"http://{self.external_addr}"


class SessionRecord(BaseModel):
    timestamp: float
    request_timestamp: float | None = None
    method: str
    path: str
    request: dict
    response: dict
    status_code: int
    # Background read of the response's output-store bundle (replay_reads); never serialized.
    _replay_read: asyncio.Future | None = PrivateAttr(default=None)

    @property
    def replay_read(self) -> asyncio.Future | None:
        return self._replay_read

    def attach_replay_read(self, read: asyncio.Future | None) -> None:
        self._replay_read = read

    def replay_outputs(self) -> "ReplayOutputs | None":
        """The arrays its output-store bundle held; valid once ``wait_for_replay_reads`` returned."""
        return None if self._replay_read is None else self._replay_read.result()


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)
