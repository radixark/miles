from __future__ import annotations

from typing import Any, Literal

from miles.utils.pydantic_utils import StrictBaseModel

HEALTH_PATH = "/v1/health"
CALL_STATUS_PATH = "/v1/calls/{call_id}"
SUBMIT_PATH = "/v1/{method_name}"

DEFAULT_POLL_TIMEOUT_SECONDS = 30.0


class SubmitRequest(StrictBaseModel):
    call_id: str
    query: dict[str, Any]


class SubmitResponse(StrictBaseModel):
    status: Literal["submitted"] = "submitted"


class CallStatusResponse(StrictBaseModel):
    status: Literal["pending", "success", "failed"]
    result: Any = None
    error: str | None = None


class HealthResponse(StrictBaseModel):
    status: Literal["ok"] = "ok"
