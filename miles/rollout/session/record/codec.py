# doc-dev: docs/developer/session_server/00-disk_offload.md

import json
import math

from miles.rollout.session.types import SessionRecord


def freeze_record(record: SessionRecord) -> SessionRecord:
    """Detach mutable containers before publishing a record to the store."""
    return record.model_copy(deep=True)


def _check_json(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Session record JSON keys must be strings")
            _check_json(item)
    elif isinstance(value, list):
        for item in value:
            _check_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Session record JSON numbers must be finite")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise TypeError(f"Unsupported session record JSON value: {type(value).__name__}")


def encode_record(record: SessionRecord) -> bytes:
    _check_json(record.request)
    _check_json(record.response)
    data = record.model_dump(mode="python")
    _check_json(data)
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def decode_record(payload: bytes) -> SessionRecord:
    data = json.loads(payload)
    if not isinstance(data, dict) or set(data) != set(SessionRecord.model_fields):
        raise ValueError("Invalid session record fields")
    _check_json(data)
    return SessionRecord.model_validate(data, strict=True)
