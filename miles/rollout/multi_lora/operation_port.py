import asyncio
from typing import Protocol

import ray

from miles.utils.operation_contract import BindingT, RegistrationKey


class OperationQueuePort(Protocol[BindingT]):
    """Ledger claims: ready_streams lists READY streams (head kind unknown); claim_data claim-and-binds in one call."""

    async def ready_streams(self) -> dict: ...

    async def claim_data(self, key: RegistrationKey) -> dict | None: ...

    async def fail(self, operation_id: str, error: str, category: str) -> None: ...


class BatchResidencyPort(Protocol[BindingT]):
    """Async transport face of controller-side TrainerResidencyPort: one immutable dispatch receipt per selection."""

    async def acquire_batch(self, bindings_by_operation: list) -> object: ...


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
