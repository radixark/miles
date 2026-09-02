from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

RegistrationKey = tuple[str, str]

BindingT = TypeVar("BindingT")


@dataclass(frozen=True)
class BatchExecutionLease(Generic[BindingT]):
    """Immutable logical-operation to physical-binding receipt for one batch."""

    dispatch_id: str
    bindings_by_operation: tuple[tuple[str, BindingT], ...]

    def binding_of(self, operation_id: str) -> BindingT | None:
        for op_id, binding in self.bindings_by_operation:
            if op_id == operation_id:
                return binding
        return None


class TrainerResidencyPort(Protocol[BindingT]):
    """Resolve, snapshot, validate, and release opaque trainer bindings."""

    def binding_for(self, key: RegistrationKey) -> BindingT | None:
        """Return the key's dispatchable binding, or ``None``."""
        ...

    def acquire_batch(self, bindings_by_operation: tuple[tuple[str, BindingT], ...]) -> BatchExecutionLease[BindingT]:
        """Snapshot validated bindings into one immutable dispatch receipt."""
        ...

    def validate(self, lease: BatchExecutionLease[BindingT]) -> bool:
        """Re-check a receipt before physical mutation."""
        ...

    def release_batch(self, lease: BatchExecutionLease[BindingT]) -> None:
        """Release any physical reservation represented by ``lease``."""
        ...


class EmptyBatchTimeoutError(RuntimeError):
    """No registration produced a claimable data operation within the wait."""
