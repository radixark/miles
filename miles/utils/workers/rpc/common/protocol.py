from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Literal

from miles.utils.pydantic_utils import StrictBaseModel

EXPECTED_BOOT_UUID_HEADER = "x-miles-expected-boot-uuid"
BOOT_UUID_HEADER = "x-miles-boot-uuid"

BOOT_UUID_MISMATCH_STATUS = 412

HEALTH_PATH = "/v1/health"
IN_FLIGHT_PATH = "/v1/in-flight"
CALL_STATUS_PATH = "/v1/calls/{call_id}"
ACKNOWLEDGE_PATH = "/v1/calls/{call_id}/ack"
SUBMIT_PATH = "/v1/{method_name}"

DEFAULT_POLL_TIMEOUT_SECONDS = 30.0

MAX_POLL_TIMEOUT_SECONDS = 300.0
MAX_AGGREGATE_REQUEST_BODY_BYTES = 64 * 1024 * 1024
MAX_CONTROL_AGGREGATE_REQUEST_BODY_BYTES = 1024 * 1024


def is_rpc_control_request(*, method: str, path: str, dynamic_paths: frozenset[str]) -> bool:
    parts = path.split("/")
    if method == "GET":
        return path in {HEALTH_PATH, IN_FLIGHT_PATH} or (
            len(parts) == 4 and parts[:3] == ["", "v1", "calls"] and bool(parts[3])
        )
    if method == "POST":
        return path in dynamic_paths or (
            len(parts) == 5 and parts[:3] == ["", "v1", "calls"] and bool(parts[3]) and parts[4] == "ack"
        )
    return False


def compute_request_digest(*, method_name: str, query: dict[str, Any]) -> bytes:
    return compute_request_identity(method_name=method_name, query=query).digest


def compute_request_identity(*, method_name: str, query: dict[str, Any]) -> RequestIdentity:
    canonical = json.dumps(
        {"method": method_name, "query": query},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return RequestIdentity(digest=hashlib.sha256(canonical).digest(), serialized_bytes=len(canonical))


@dataclasses.dataclass(frozen=True, slots=True)
class RequestIdentity:
    digest: bytes
    serialized_bytes: int


class SubmitRequest(StrictBaseModel):
    call_id: str
    query: dict[str, Any]


class SubmitResponse(StrictBaseModel):
    status: Literal["submitted"] = "submitted"


class AcknowledgeRequest(StrictBaseModel):
    request_digest: str


class AcknowledgeResponse(StrictBaseModel):
    status: Literal["acknowledged"] = "acknowledged"


class CallStatusResponse(StrictBaseModel):
    status: Literal["pending", "success", "failed"]
    result: Any = None
    error: str | None = None


class HealthResponse(StrictBaseModel):
    status: Literal["ok"] = "ok"


class InFlightResponse(StrictBaseModel):
    call_ids: list[str]
