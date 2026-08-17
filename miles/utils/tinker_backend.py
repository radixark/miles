"""Serving identity for the tinker-compatible backend.

Every engine-facing artifact carries the full registration identity: a
re-registered name is a new tenant, so nothing minted by a predecessor — a
request id, an engine-side LoRA name, a KV-cache key — can alias its
successor (anti-ABA)."""

import time
import uuid
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from miles.utils.misc import SingletonMeta

# Cannot appear in adapter names (registry validates [A-Za-z0-9._-] only).
RID_SEPARATOR = "::"

# The protocol identity of one registration of one adapter name: a
# re-registered name is a new key. Shared by claim receipts, batch commits,
# the gradient-window tracker, and physical-executor validation
# (codex-rollout-fullparameter-design-0810 §5.9).
RegistrationKey = tuple[str, str]

# Opaque execution binding: what a trainer needs to route one logical
# operation onto physical state. The Multi-LoRA concrete is ResidentBinding
# (registration -> fixed slot); a future parameterization supplies its own.
BindingT = TypeVar("BindingT")


@dataclass(frozen=True)
class BatchExecutionLease(Generic[BindingT]):
    """Immutable receipt for ONE trainer dispatch: it fixes the logical
    operation -> opaque execution binding mapping for the batch's lifetime
    (codex-rollout-fullparameter-design-0810 §5.3). ``dispatch_id`` exists for
    logging/correlation only — there is no active/released lease registry.
    The receipt lives to the operation completion boundary: a data batch to
    ``commit_tinker_batch``, immediate controls to their completion, deferred
    publish/load past the physical publish barrier."""

    dispatch_id: str
    bindings_by_operation: tuple[tuple[str, BindingT], ...]

    def binding_of(self, operation_id: str) -> BindingT | None:
        for op_id, binding in self.bindings_by_operation:
            if op_id == operation_id:
                return binding
        return None


class TrainerResidencyPort(Protocol[BindingT]):
    """Narrow facade over trainer residency: batch construction sees opaque
    bindings and batch receipts, never SlotPool internals. The current (and
    only) concrete is FixedSlotResidency — it snapshots and validates mappings
    that fixed residency already established, and never binds, unbinds, picks
    victims, or moves state. Fixed residency is a current implementation
    policy, not part of this contract (§3.8)."""

    def binding_for(self, key: RegistrationKey) -> BindingT | None:
        """The exact registration's current binding, or None when it may not
        be dispatched (the claim gate). Never mutates residency."""
        ...

    def acquire_batch(self, bindings_by_operation: tuple[tuple[str, BindingT], ...]) -> BatchExecutionLease[BindingT]:
        """Snapshot already-claimed bindings into one immutable dispatch
        receipt, re-validating ownership. Raises if any binding went stale."""
        ...

    def validate(self, lease: BatchExecutionLease[BindingT]) -> bool:
        """Re-check the receipt before physical mutation."""
        ...

    def release_batch(self, lease: BatchExecutionLease[BindingT]) -> None:
        """Lifecycle hook at the batch's completion boundary; the fixed
        residency concrete is a no-op (nothing was reserved), so failure
        paths cannot leak capacity state."""
        ...


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


def uses_tinker_operation_semantics(args) -> bool:
    """Protocol mode: the run is driven by explicit client operations, so the
    trainer keeps accumulated gradients across train calls and steps the
    optimizer only when a client optim_step executes. This is a property of
    the tinker operation protocol, not of the parameterization; validation
    currently rejects it without multi-LoRA slots, so for every launched
    config it coincides with ``uses_multi_lora_tinker_executor``
    (tests/fast/utils/test_tinker_predicates.py witnesses that equivalence)."""
    return bool(getattr(args, "tinker_backend", False))


def uses_multi_lora_tinker_executor(args) -> bool:
    """Parameter executor: tinker operations execute on multi-LoRA trainer
    slots (per-slot optimizer children, adapter routing, slot publish). The
    only executor implemented; a future full-parameter executor would satisfy
    ``uses_tinker_operation_semantics`` without this predicate."""
    return uses_tinker_operation_semantics(args) and getattr(args, "multi_lora_n_adapters", 0) > 0


def is_tinker_enabled(args) -> bool:
    """Tinker mode: multi-LoRA slots driven by the tinker operation backend."""
    return uses_multi_lora_tinker_executor(args)


def validate_tinker_args(args) -> None:
    """Default and validate the tinker arg surface (after the shared multi-LoRA
    validation). Tinker replaces the dataset rollout plane: operations carry
    the data, so the rollout fn and data source swap to the queue-driven pair."""
    if not getattr(args, "tinker_backend", False):
        # The frontend flags ride on the backend; alone they would silently
        # no-op (no frontend starts, the key guards nothing) — fail loud.
        assert not getattr(args, "tinker_frontend", False), "--tinker-frontend requires --tinker-backend"
        assert not getattr(args, "tinker_api_key", None), "--tinker-api-key requires --tinker-frontend"
        return
    assert not (
        getattr(args, "tinker_api_key", None) and not getattr(args, "tinker_frontend", False)
    ), "--tinker-api-key requires --tinker-frontend (only the SDK frontend authenticates requests)"
    from miles.utils.environ import enable_experimental_rollout_refactor

    assert getattr(args, "multi_lora_n_adapters", 0) > 0, "--tinker-backend requires --multi-lora-n-adapters > 0"
    assert enable_experimental_rollout_refactor(), (
        "--tinker-backend needs the class-based rollout API: set MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 "
        "(and propagate it through runtime_env when submitting via Ray)"
    )
    if getattr(args, "tinker_frontend", False) and not getattr(args, "multi_lora_http_server_path", None):
        args.multi_lora_http_server_path = "miles.ray.tinker_backend.frontend.http_server.TinkerFrontendHTTPServer"
    if args.rollout_function_path is None:
        args.rollout_function_path = "miles.rollout.tinker_backend.rollout_fn.TinkerRolloutFn"
    if args.data_source_path == "miles.rollout.data_source.RolloutDataSourceWithBuffer":
        args.data_source_path = "miles.rollout.tinker_backend.rollout_fn.TinkerNullDataSource"
    # One selection = one whole train step: the multi-LoRA dynamic-GBS branch
    # sizes the step to the (zero-weight padded) batch, so trimming is a
    # structural no-op.
    args.use_dynamic_global_batch_size = True
