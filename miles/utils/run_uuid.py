import re
import uuid

RUN_UUID_LENGTH = 16

_RUN_UUID_PATTERN = re.compile(rf"[0-9a-f]{{{RUN_UUID_LENGTH}}}")


def generate_run_uuid() -> str:
    return uuid.uuid4().hex[:RUN_UUID_LENGTH]


def validate_run_uuid(value: str) -> str:
    if _RUN_UUID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"invalid run uuid {value!r}; expected exactly {RUN_UUID_LENGTH} lowercase hex characters")
    return value
