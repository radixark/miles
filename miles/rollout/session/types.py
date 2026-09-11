from pydantic import BaseModel, Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SessionServerInstance(FrozenStrictBaseModel):
    """One session-server instance as the driver published it.

    A pick from ``args.session_server_instances`` returns everything about the
    instance at once, so nothing has to stay positionally aligned across
    separate structures.
    """

    # ``host:port`` the driver dials.
    addr: str
    # ``host:port`` a peer outside the cluster dials; defaults to ``addr``, which
    # is what a deployment whose node addresses already route from outside wants.
    external_addr: str = None  # type: ignore[assignment]  # filled by the validator below
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


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)
