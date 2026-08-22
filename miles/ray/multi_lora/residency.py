import logging
import uuid
from dataclasses import dataclass

from miles.ray.multi_lora.registry import AdapterRegistry, AdapterState
from miles.utils.operation_contract import BatchExecutionLease, RegistrationKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResidentBinding:
    """Multi-LoRA execution binding: one registration pinned to its fixed
    trainer slot. Opaque above the residency port — batch plumbing forwards
    it, only Multi-LoRA code interprets it."""

    registration_key: RegistrationKey
    training_slot: int


class FixedSlotResidency:
    """TrainerResidencyPort[ResidentBinding] over the adapter registry."""

    def __init__(self, registry: AdapterRegistry) -> None:
        self.registry = registry

    def binding_for(self, key: RegistrationKey) -> ResidentBinding | None:
        name, registration_id = key
        record = self.registry.find(name)
        if (
            record is None
            or record.registration_id != registration_id
            or record.state is not AdapterState.READY
            or record.slot is None
        ):
            return None
        return ResidentBinding(registration_key=key, training_slot=record.slot)

    def acquire_batch(
        self, bindings_by_operation: tuple[tuple[str, ResidentBinding], ...]
    ) -> BatchExecutionLease[ResidentBinding]:
        for operation_id, binding in bindings_by_operation:
            if not self._owns_slot(binding):
                raise ValueError(
                    f"operation '{operation_id}': registration "
                    f"{binding.registration_key} no longer owns trainer slot {binding.training_slot}"
                )
        return BatchExecutionLease(
            dispatch_id=uuid.uuid4().hex,
            bindings_by_operation=tuple(bindings_by_operation),
        )

    def release_batch(self, lease: BatchExecutionLease[ResidentBinding]) -> None:
        """No-op lifecycle hook (nothing to free under fixed residency)."""

    def _owns_slot(self, binding: ResidentBinding) -> bool:
        name, registration_id = binding.registration_key
        record = self.registry.records.get(name)
        return (
            record is not None
            and record.registration_id == registration_id
            and record.slot == binding.training_slot
            and record.state in (AdapterState.READY, AdapterState.RETIRING)
        )


# ---------------- data-plane encoding ----------------
# The lease crosses the rollout -> object store -> trainer boundary as plain
# data (the store's codecs never see a dataclass); typed leases live at the
# controller/adapter boundaries.


def lease_to_metadata(lease: BatchExecutionLease[ResidentBinding]) -> dict:
    return {
        "dispatch_id": lease.dispatch_id,
        "bindings_by_operation": [
            [op_id, [binding.registration_key[0], binding.registration_key[1], binding.training_slot]]
            for op_id, binding in lease.bindings_by_operation
        ],
    }


def lease_from_metadata(data: dict) -> BatchExecutionLease[ResidentBinding]:
    return BatchExecutionLease(
        dispatch_id=data["dispatch_id"],
        bindings_by_operation=tuple(
            (op_id, ResidentBinding(registration_key=(name, registration_id), training_slot=slot))
            for op_id, (name, registration_id, slot) in data["bindings_by_operation"]
        ),
    )
