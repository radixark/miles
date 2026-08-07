"""Per-registration operation ledger for the tinker backend.

Clients push protocol-neutral operations; data-bearing kinds ride the rollout
selection path through the queue child rollout fn, data-less kinds execute in
the driver's control phase. One registration is strictly serialized: an
operation is claimable only when every earlier operation reached a terminal
state, which carries the client's per-model ordering end to end.

Arrival may be OUT OF ORDER (the tinker SDK deliberately posts the first
chunk of a large forward_backward last): operations buffer by ordinal and a
gap below the head blocks claims until it fills. Ordinals are consecutive
integers starting at 1 per registration.
NOTE(frontend): when a tinker HTTP frontend lands, this arrival
reorder/gap-buffer moves there ((model_id, seq_id) reordering); the backend
then reverts to strictly-increasing arrival.

Retries are fingerprinted: re-enqueueing a known operation_id with an
identical (kind, payload) returns the original operation; a different
fingerprint is a conflict error, never silently swallowed.

All mutations run inside the controller actor between awaits, so ledger
methods are synchronous and atomic by construction.
"""

import hashlib
import json
import logging
from bisect import insort
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

Tenant = tuple[str, str]


class OperationKind(str, Enum):
    FORWARD_BACKWARD = "forward_backward"
    FORWARD = "forward"
    OPTIM_STEP = "optim_step"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"
    SAVE_STATE = "save_state"
    LOAD_STATE = "load_state"


# Ride the rollout/BatchPlan path (they carry Datums).
DATA_KINDS = frozenset({OperationKind.FORWARD_BACKWARD, OperationKind.FORWARD})
# Execute in the driver's control phase (no Datums).
CONTROL_KINDS = frozenset(OperationKind) - DATA_KINDS


class OperationState(str, Enum):
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset({OperationState.SUCCEEDED, OperationState.FAILED, OperationState.CANCELLED})


class OperationBackpressure(RuntimeError):
    """Queue or unacked-result capacity reached; the caller must retry later
    (the HTTP layer maps this to 429 + Retry-After — 4xx families the tinker
    SDK treats as fatal must never carry backpressure)."""


def payload_fingerprint(kind: str, payload: dict | None) -> str:
    """Canonical digest of an operation's identity-relevant content."""
    canonical = json.dumps({"kind": kind, "payload": payload or {}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class Operation:
    operation_id: str
    name: str
    registration_id: str
    # Consecutive from 1 per registration; arrival may be out of order.
    ordinal: int
    kind: OperationKind
    payload: dict = field(default_factory=dict)
    fingerprint: str = ""
    state: OperationState = OperationState.QUEUED
    result: dict | None = None
    error: str | None = None
    # "user" (bad request / cancelled by lifecycle) or "server" (execution failure).
    error_category: str | None = None

    @property
    def tenant(self) -> Tenant:
        return (self.name, self.registration_id)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def view(self) -> dict:
        return dict(
            operation_id=self.operation_id,
            name=self.name,
            registration_id=self.registration_id,
            ordinal=self.ordinal,
            kind=self.kind.value,
            state=self.state.value,
            result=self.result,
            error=self.error,
            error_category=self.error_category,
        )

    def claimed_view(self) -> dict:
        """Executor-facing view: the request payload rides only on claims
        (forward_backward samples, adam_params, save/load targets) so poll
        results stay lean."""
        return {**self.view(), "payload": self.payload}


@dataclass
class _RegistrationQueue:
    """Ordinal-sorted operations of one registration, pending and terminal."""

    operations: list[Operation] = field(default_factory=list)
    by_ordinal: dict[int, Operation] = field(default_factory=dict)
    fenced: bool = False

    def insert(self, op: Operation) -> None:
        insort(self.operations, op, key=lambda o: o.ordinal)
        self.by_ordinal[op.ordinal] = op

    def contiguous_arrived(self) -> int:
        """Largest K such that ordinals 1..K have all arrived."""
        k = 0
        while (k + 1) in self.by_ordinal:
            k += 1
        return k

    def first_open(self) -> Operation | None:
        """The lowest-ordinal non-terminal operation, only when no arrival
        gap sits below it (strict execution order despite unordered arrival)."""
        for op in self.operations:
            if not op.terminal:
                return op if op.ordinal <= self.contiguous_arrived() else None
        return None

    def open_count(self) -> int:
        return sum(1 for op in self.operations if not op.terminal)

    def unacked_terminal_count(self) -> int:
        return sum(1 for op in self.operations if op.terminal)


class OperationLedger:
    """All registrations' queues plus the operation_id index."""

    def __init__(self, max_pending: int = 256, max_unacked_results: int = 4096) -> None:
        self.max_pending = max_pending
        self.max_unacked_results = max_unacked_results
        self.queues: dict[Tenant, _RegistrationQueue] = {}
        self.by_id: dict[str, Operation] = {}

    # ------------------------------ enqueue ------------------------------

    def enqueue(
        self,
        operation_id: str,
        name: str,
        registration_id: str,
        ordinal: int,
        kind: str,
        payload: dict | None = None,
    ) -> dict:
        """Buffer one operation; idempotent on (operation_id, fingerprint)."""
        fingerprint = payload_fingerprint(kind, payload)
        if (existing := self.by_id.get(operation_id)) is not None:
            if existing.fingerprint != fingerprint or existing.tenant != (name, registration_id):
                raise ValueError(
                    f"operation '{operation_id}' already exists with different content; "
                    "retries must resend the identical request"
                )
            return existing.view()

        queue = self.queues.setdefault((name, registration_id), _RegistrationQueue())
        if queue.fenced:
            raise ValueError(f"registration '{name}' ({registration_id[:8]}) is retired; operations are fenced")
        if ordinal < 1:
            raise ValueError(f"operation '{operation_id}' ordinal must be >= 1, got {ordinal}")
        if (holder := queue.by_ordinal.get(ordinal)) is not None:
            raise ValueError(
                f"ordinal {ordinal} already taken by operation '{holder.operation_id}'; "
                "per-registration ordinals are unique and consecutive"
            )
        if queue.open_count() >= self.max_pending:
            raise OperationBackpressure(f"registration '{name}' has {self.max_pending} operations pending")
        if queue.unacked_terminal_count() >= self.max_unacked_results:
            raise OperationBackpressure(
                f"registration '{name}' holds {self.max_unacked_results} unacknowledged results; ack or deregister"
            )

        op = Operation(
            operation_id=operation_id,
            name=name,
            registration_id=registration_id,
            ordinal=ordinal,
            kind=OperationKind(kind),
            payload=payload or {},
            fingerprint=fingerprint,
        )
        queue.insert(op)
        self.by_id[operation_id] = op
        return op.view()

    # ------------------------------ claims ------------------------------

    def claim_data_operation(self, name: str, registration_id: str) -> dict | None:
        """Claim the registration's next operation when it is data-bearing.
        Strict serialization: nothing is claimable while an earlier operation
        is open or missing, so an optim_step never overtakes its batches."""
        queue = self.queues.get((name, registration_id))
        if queue is None:
            return None
        op = queue.first_open()
        if op is None or op.state is not OperationState.QUEUED or op.kind not in DATA_KINDS:
            return None
        op.state = OperationState.CLAIMED
        return op.claimed_view()

    def claimable_control_tenants(self) -> list[Tenant]:
        """Registrations whose next open operation is a control kind (the
        caller filters by adapter state/slot residency before claiming)."""
        tenants = []
        for tenant, queue in self.queues.items():
            op = queue.first_open()
            if op is not None and op.state is OperationState.QUEUED and op.kind in CONTROL_KINDS:
                tenants.append(tenant)
        return tenants

    def claim_control_operation(
        self, name: str, registration_id: str, kinds: tuple[str, ...] | None = None
    ) -> dict | None:
        queue = self.queues.get((name, registration_id))
        if queue is None:
            return None
        op = queue.first_open()
        if op is None or op.state is not OperationState.QUEUED or op.kind not in CONTROL_KINDS:
            return None
        if kinds is not None and op.kind.value not in kinds:
            return None
        op.state = OperationState.CLAIMED
        return op.claimed_view()

    # ------------------------------ terminals ------------------------------

    def complete(self, operation_id: str, result: dict | None = None) -> None:
        op = self._open_op(operation_id)
        op.state = OperationState.SUCCEEDED
        op.result = result

    def fail(self, operation_id: str, error: str, category: str = "server") -> None:
        op = self._open_op(operation_id)
        op.state = OperationState.FAILED
        op.error = error
        op.error_category = category

    def cancel(self, operation_id: str) -> dict:
        """Cancel a not-yet-claimed operation; anything already claimed must
        run to a terminal state (a half-executed optimizer mutation cannot be
        rolled back). A cancelled ordinal still counts for contiguity."""
        op = self.by_id.get(operation_id)
        if op is None:
            raise KeyError(f"unknown operation '{operation_id}'")
        if op.state is not OperationState.QUEUED:
            raise ValueError(f"operation '{operation_id}' is {op.state.value}; only QUEUED operations cancel")
        op.state = OperationState.CANCELLED
        op.error = "cancelled by client"
        op.error_category = "user"
        return op.view()

    def _open_op(self, operation_id: str) -> Operation:
        op = self.by_id.get(operation_id)
        if op is None:
            raise KeyError(f"unknown operation '{operation_id}'")
        if op.terminal:
            raise ValueError(f"operation '{operation_id}' already terminal ({op.state.value})")
        return op

    # ------------------------------ results ------------------------------

    def get(self, operation_id: str) -> dict | None:
        op = self.by_id.get(operation_id)
        return op.view() if op is not None else None

    def ack(self, operation_id: str) -> None:
        """Drop a terminal record the client has retrieved. Terminal records
        are never evicted by pressure while their registration lives — the
        enqueue backpressure cap is the knob, not result eviction. The
        ordinal stays reserved for contiguity."""
        op = self.by_id.get(operation_id)
        if op is None:
            return
        if not op.terminal:
            raise ValueError(f"operation '{operation_id}' is {op.state.value}; ack applies to terminal operations")
        self.by_id.pop(operation_id, None)
        queue = self.queues.get(op.tenant)
        if queue is not None:
            queue.operations = [o for o in queue.operations if o.operation_id != operation_id]
            # by_ordinal keeps the slot so contiguity and ordinal uniqueness survive the ack.
            if not queue.operations and queue.fenced:
                self.queues.pop(op.tenant, None)

    # ------------------------------ fencing ------------------------------

    def fence(self, name: str, registration_id: str) -> list[str]:
        """Terminal-fail every open operation of a dead registration and
        refuse new ones. Terminal records stay retrievable until acked."""
        queue = self.queues.get((name, registration_id))
        if queue is None or queue.fenced:
            return []
        queue.fenced = True
        failed = []
        for op in queue.operations:
            if not op.terminal:
                op.state = OperationState.FAILED
                op.error = "registration retired before the operation ran"
                op.error_category = "user"
                failed.append(op.operation_id)
        return failed

    def queue_view(self, name: str, registration_id: str) -> list[dict]:
        queue = self.queues.get((name, registration_id))
        return [op.view() for op in queue.operations] if queue is not None else []
