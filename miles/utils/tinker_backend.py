"""Serving identity for the tinker-compatible backend.

Every engine-facing artifact carries the full registration identity: a
re-registered name is a new tenant, so nothing minted by a predecessor — a
request id, an engine-side LoRA name, a KV-cache key — can alias its
successor (anti-ABA)."""

import time
import uuid
from dataclasses import dataclass

from miles.utils.misc import SingletonMeta

# Cannot appear in adapter names (registry validates [A-Za-z0-9._-] only).
RID_SEPARATOR = "::"


class AdaptersCache(metaclass=SingletonMeta):
    """TTL-cached tinker controller snapshot; get/get_all expose the resident
    projection (ready + retiring), used by the generate path to drop requests
    for adapters that are no longer served."""

    def __init__(self, ttl_s: float = 1.0) -> None:
        self.ttl_s = ttl_s
        self.snapshot: dict = {"pending": {}, "ready": {}, "retiring": {}, "cleanup": []}
        self.last_refresh: float | None = None

    async def get_snapshot(self) -> dict:
        from miles.ray.tinker_backend.controller import get_tinker_controller

        now = time.monotonic()
        if self.last_refresh is None or now - self.last_refresh >= self.ttl_s:
            try:
                self.snapshot = await get_tinker_controller().snapshot.remote()
                self.last_refresh = now
            except Exception:
                pass
        return self.snapshot

    async def get_all(self) -> dict:
        snapshot = await self.get_snapshot()
        return {**snapshot.get("ready", {}), **snapshot.get("retiring", {})}

    async def get(self, adapter_name: str):
        return (await self.get_all()).get(adapter_name)


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


def is_tinker_enabled(args) -> bool:
    """Tinker mode: multi-LoRA slots driven by the tinker operation backend."""
    return bool(getattr(args, "tinker_backend", False)) and getattr(args, "multi_lora_n_adapters", 0) > 0


def validate_tinker_args(args) -> None:
    """Default and validate the tinker arg surface (after the shared multi-LoRA
    validation). Tinker replaces the dataset rollout plane: operations carry
    the data, so the rollout fn and data source swap to the queue-driven pair."""
    if not getattr(args, "tinker_backend", False):
        return
    from miles.utils.environ import enable_experimental_rollout_refactor

    assert getattr(args, "multi_lora_n_adapters", 0) > 0, "--tinker-backend requires --multi-lora-n-adapters > 0"
    assert enable_experimental_rollout_refactor(), (
        "--tinker-backend needs the class-based rollout API: set MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 "
        "(and propagate it through runtime_env when submitting via Ray)"
    )
    if args.rollout_function_path is None:
        args.rollout_function_path = "miles.rollout.tinker_backend.rollout_fn.TinkerRolloutFn"
    if args.data_source_path == "miles.rollout.data_source.RolloutDataSourceWithBuffer":
        args.data_source_path = "miles.rollout.tinker_backend.rollout_fn.TinkerNullDataSource"
    # One selection = one whole train step: the multi-LoRA dynamic-GBS branch
    # sizes the step to the (zero-weight padded) batch, so trimming is a
    # structural no-op.
    args.use_dynamic_global_batch_size = True
