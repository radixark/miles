from pydantic import BaseModel, Field

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class SessionServerInstance(FrozenStrictBaseModel):
    """One session-server instance as the driver published it.

    A pick from ``args.session_server_instances`` returns everything about the
    instance at once, so nothing has to stay positionally aligned across
    separate structures.
    """

    # ``host:port`` the driver dials.
    addr: str
    instance_id: str | None = None

    @property
    def url(self) -> str:
        return f"http://{self.addr}"


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
