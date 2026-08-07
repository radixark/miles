"""Serving identity for the tinker-compatible backend.

Every engine-facing artifact carries the full registration identity: a
re-registered name is a new tenant, so nothing minted by a predecessor — a
request id, an engine-side LoRA name, a KV-cache key — can alias its
successor (anti-ABA)."""

import uuid
from dataclasses import dataclass

# Cannot appear in adapter names (registry validates [A-Za-z0-9._-] only).
RID_SEPARATOR = "::"


@dataclass(frozen=True)
class TinkerAdapterRef:
    """Stamp on every sample a tinker run emits: routing derives from
    ``(name, registration_id)``; ``slot`` is trainer-side only."""

    name: str
    registration_id: str
    serving_version: int
    slot: int | None


class EmptyBatchTimeoutError(RuntimeError):
    """No registration produced a claimable data operation within the wait."""


def make_rid(adapter_name: str, registration_id: str) -> str:
    """Request id carrying the full registration: a stale tenant's prefix abort
    can never match a same-name successor's requests."""
    return f"{adapter_name}{RID_SEPARATOR}{registration_id}{RID_SEPARATOR}{uuid.uuid4().hex}"


def rid_prefix(adapter_name: str, registration_id: str) -> str:
    """Abort-by-prefix namespace for one registration of one adapter."""
    return f"{adapter_name}{RID_SEPARATOR}{registration_id}{RID_SEPARATOR}"


def parse_adapter(rid: str) -> str:
    # The separator cannot appear in adapter names, so the first segment is the name.
    return rid.split(RID_SEPARATOR, 1)[0]


def serving_lora_name(adapter_name: str, registration_id: str) -> str:
    """Engine-side LoRA name for one registration; pushes and every inference
    request must agree on it, and a re-registered name is a new tenant."""
    return f"__miles_adapter_{adapter_name}_{registration_id}"


def cache_extra_key(adapter_name: str, registration_id: str, serving_version: int) -> str:
    """KV-cache namespace: registration and serving version both enter the key, so
    neither a re-registered name nor a republished revision can reuse stale KV."""
    return f"{adapter_name}:{registration_id}:v{serving_version}"
