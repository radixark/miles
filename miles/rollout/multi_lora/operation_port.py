"""Operation-queue and residency transports for Multi-LoRA operation batches
(codex-rollout-fullparameter-design-0810 §4.5).

The adapter's scheduling logic (RR, coalesce, kind lock, whole-batch
selection) talks to these narrow ports; ONLY the Ray concretes below know
``get_multi_lora_controller()``, ``.remote()`` and ``ray.get`` — a future
RolloutExecutor injects its own transports and the adapter's policy code
never changes, and unit tests drive the scheduler with fakes instead of a
Ray cluster."""

import asyncio
from typing import Protocol

import ray

from miles.utils.operation_contract import BindingT, RegistrationKey


class OperationQueuePort(Protocol[BindingT]):
    """Claims against the backend's operation ledger.

    ``ready_streams`` lists the current READY registration streams (keyed by
    name, valued by the controller's run views) — these are streams, not
    unclaimed operation candidates: a stream's head kind is unknown until
    claimed. ``claim_data`` is claim-and-bind in ONE backend actor call: the
    exact READY binding resolves first, only then does the ledger turn the
    head CLAIMED, and the returned claim carries the binding; a missing
    binding leaves the head QUEUED."""

    async def ready_streams(self) -> dict: ...

    async def claim_data(self, key: RegistrationKey) -> dict | None: ...

    async def fail(self, operation_id: str, error: str, category: str) -> None: ...


class BatchResidencyPort(Protocol[BindingT]):
    """Selection-side view of the trainer-residency facade: after RR/coalesce
    picks a selection, acquire ONE immutable dispatch receipt for its
    already-claimed bindings. (The synchronous port lives controller-side —
    miles/utils/operation_contract.TrainerResidencyPort; this is its async
    transport face.)"""

    async def acquire_batch(self, bindings_by_operation: list) -> object: ...

    async def release_batch(self, lease: object) -> None: ...


class BatchAbortPort(Protocol):
    """Finalize a claimed operation batch that will not reach the trainer."""

    async def abort_batch(self, operation_ids: list[str], error: str, lease_metadata: dict | None) -> None: ...


class RayMultiLoraOperationQueue:
    """Only this class (and its residency sibling) knows the Ray controller,
    .remote(), and ray.get."""

    async def ready_streams(self) -> dict:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        snapshot = await asyncio.to_thread(ray.get, get_multi_lora_controller().snapshot.remote())
        return snapshot["ready"]

    async def claim_data(self, key: RegistrationKey) -> dict | None:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        name, registration_id = key
        return await asyncio.to_thread(
            ray.get, get_multi_lora_controller().claim_data_operation.remote(name, registration_id)
        )

    async def fail(self, operation_id: str, error: str, category: str) -> None:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        await asyncio.to_thread(
            ray.get, get_multi_lora_controller().fail_operation.remote(operation_id, error, category)
        )


class RayTrainerResidencyPort:
    """Thin async proxy to the backend-owned FixedSlotResidency."""

    async def acquire_batch(self, bindings_by_operation: list) -> object:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        return await asyncio.to_thread(
            ray.get, get_multi_lora_controller().acquire_batch_lease.remote(list(bindings_by_operation))
        )

    async def release_batch(self, lease: object) -> None:
        """Release the typed lease even if plain receipt encoding failed."""
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        await asyncio.to_thread(ray.get, get_multi_lora_controller().release_batch_lease.remote(lease))


class RayMultiLoraBatchAbort:
    """BatchAbortPort concrete over the controller's idempotent finalizer."""

    async def abort_batch(self, operation_ids: list[str], error: str, lease_metadata: dict | None) -> None:
        from miles.ray.multi_lora.controller import get_multi_lora_controller

        await asyncio.to_thread(
            ray.get,
            get_multi_lora_controller().fail_tinker_batch.remote(list(operation_ids), error, lease_metadata),
        )
