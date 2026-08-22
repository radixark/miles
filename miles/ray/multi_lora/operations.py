import hashlib
import json
import logging
import time
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
class SealedGap:
    """Contiguity filler for an ordinal whose submission never arrived within
    the gap timeout. The tinker SDK can consume a seq_id and then fail BEFORE
    HTTP (non-finite JSON serialization, a cancelled future): no retry will
    ever fill that ordinal, so the seal restores liveness without relaxing
    the fence — the missing ordinal's identity is never executed (a late
    genuine arrival hits the ordinal-taken conflict), its kind is never
    guessed, and the poison scan treats the seal as neutral (it contributed
    no gradients and delimits no window)."""

    operation_id: str
    ordinal: int


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
    was_claimed: bool = False
    window_consumed: bool = False
    # Monotonic stamp of the QUEUED->CLAIMED transition; the claimed-TTL sweep ages against it.
    claimed_at: float | None = None

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
    by_ordinal: dict[int, "Operation | SealedGap"] = field(default_factory=dict)
    fenced: bool = False
    # Cached contiguity frontier; ordinals are never removed, so it only advances.
    _contiguous: int = 0
    # Gap-stall clock: the missing ordinal the queue is blocked on and when
    # that block was first observed. A different hole restarts the clock.
    _stall_missing: int | None = None
    _stall_since: float | None = None

    def insert(self, op: Operation) -> None:
        insort(self.operations, op, key=lambda o: o.ordinal)
        self.by_ordinal[op.ordinal] = op

    def contiguous_arrived(self) -> int:
        """Largest K such that ordinals 1..K have all arrived."""
        k = self._contiguous
        while (k + 1) in self.by_ordinal:
            k += 1
        self._contiguous = k
        return k

    def fills_blocking_gap(self, ordinal: int) -> bool:
        if not self.operations or ordinal >= self.operations[-1].ordinal:
            return False
        return ordinal == self.contiguous_arrived() + 1

    def first_open(self) -> Operation | None:
        for op in self.operations:
            if not op.terminal:
                return op if op.ordinal <= self.contiguous_arrived() else None
        return None

    def open_count(self) -> int:
        return sum(1 for op in self.operations if not op.terminal)

    def unacked_terminal_count(self) -> int:
        return sum(1 for op in self.operations if op.terminal)

    def gap_stall(self, now: float) -> tuple[int, float] | None:
        if self.fenced or self.first_open() is not None or self.open_count() == 0:
            self._stall_missing = self._stall_since = None
            return None
        missing = self.contiguous_arrived() + 1
        if self._stall_missing != missing:
            self._stall_missing, self._stall_since = missing, now
        return missing, now - self._stall_since


class OperationLedger:
    """All registrations' queues plus the operation_id index."""

    def __init__(
        self,
        max_pending: int = 256,
        max_unacked_results: int = 4096,
        gap_timeout: float | None = 600.0,
        claimed_ttl: float | None = 1800.0,
        time_fn=time.monotonic,
    ) -> None:
        self.max_pending = max_pending
        self.max_unacked_results = max_unacked_results
        self.gap_timeout = gap_timeout
        self.claimed_ttl = claimed_ttl
        self._time = time_fn
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
            if (
                existing.fingerprint != fingerprint
                or existing.tenant != (name, registration_id)
                or existing.ordinal != ordinal
            ):
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
        if queue.open_count() >= self.max_pending and not queue.fills_blocking_gap(ordinal):
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
        queue = self.queues.get((name, registration_id))
        if queue is None:
            return None
        op = queue.first_open()
        if op is None or op.state is not OperationState.QUEUED or op.kind not in DATA_KINDS:
            return None
        op.state = OperationState.CLAIMED
        op.was_claimed = True
        op.claimed_at = self._time()
        return op.claimed_view()

    def claimable_control_tenants(self) -> list[Tenant]:
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
        op.was_claimed = True
        op.claimed_at = self._time()
        return op.claimed_view()

    def poisoned_window_blocker(self, name: str, registration_id: str, ordinal: int) -> str | None:
        queue = self.queues.get((name, registration_id))
        if queue is None:
            return None
        for o in range(ordinal - 1, 0, -1):
            op = queue.by_ordinal.get(o)
            if op is None or isinstance(op, SealedGap):
                continue
            if op.kind is OperationKind.OPTIM_STEP and op.was_claimed and op.terminal and op.window_consumed:
                return None
            if op.kind is OperationKind.FORWARD_BACKWARD and op.terminal and op.state is not OperationState.SUCCEEDED:
                return f"forward_backward ordinal {o} {op.state.value}: {op.error or 'failed'}"
        return None

    # ------------------------------ gap stalls ------------------------------
    # A client can consume an ordinal and then fail BEFORE HTTP (the 0.24.1
    # SDK serializes AFTER taking its seq counter: non-finite floats raise a
    # local ValueError, an immediately-cancelled future never posts). No retry
    # fills such a hole, so the buffered tail would wait forever. Enforcement
    # never relaxes the fence: nothing is skipped, no kind is guessed, no
    # operation runs out of order — the blocked (never-claimed) operations
    # terminal-fail typed and the hole is sealed against late execution.

    def gap_stalls(self, now: float | None = None) -> list[dict]:
        """Current stalls (observability): registrations whose open operations
        are all buffered above an arrival hole, with the hole's ordinal, its
        age, and the number of operations blocked behind it."""
        now = self._time() if now is None else now
        stalls = []
        for (name, registration_id), queue in self.queues.items():
            stall = queue.gap_stall(now)
            if stall is not None:
                missing, stalled_for = stall
                stalls.append(
                    dict(
                        name=name,
                        registration_id=registration_id,
                        missing_ordinal=missing,
                        stalled_for=stalled_for,
                        blocked_operations=queue.open_count(),
                    )
                )
        return stalls

    def sweep_gap_timeouts(self, now: float | None = None) -> list[dict]:
        now = self._time() if now is None else now
        if self.gap_timeout is None or self.gap_timeout <= 0:
            for queue in self.queues.values():  # keep stall clocks observable
                queue.gap_stall(now)
            return []
        events = []
        for stall in self.gap_stalls(now):
            if stall["stalled_for"] >= self.gap_timeout:
                events.append(self._expire_stall(stall))
        return events

    def _expire_stall(self, stall: dict) -> dict:
        queue = self.queues[(stall["name"], stall["registration_id"])]
        missing, stalled_for = stall["missing_ordinal"], stall["stalled_for"]
        last_arrived = max(queue.by_ordinal)
        sealed = []
        for ordinal in range(missing, last_arrived):
            if ordinal not in queue.by_ordinal:
                queue.by_ordinal[ordinal] = SealedGap(
                    operation_id=f"{stall['name']}:gap-sealed:{ordinal}", ordinal=ordinal
                )
                sealed.append(ordinal)
        failed = []
        for op in queue.operations:
            if not op.terminal:  # all QUEUED: nothing is claimable while the queue stalls
                op.state = OperationState.FAILED
                op.error = (
                    f"operation gap timeout: ordinal {op.ordinal} waited {stalled_for:.0f}s behind missing "
                    f"ordinal {missing}, whose submission never reached the server (it failed client-side "
                    "before HTTP — e.g. non-finite values failing JSON serialization, or a cancelled SDK "
                    "future); the never-arrived ordinals are sealed and will never execute — resubmit this "
                    "work as new operations"
                )
                op.error_category = "user"
                failed.append(op.operation_id)
        queue._stall_missing = queue._stall_since = None
        event = {**stall, "sealed_ordinals": sealed, "failed_operations": failed}
        logger.warning(
            f"[tinker] gap timeout on '{stall['name']}' ({stall['registration_id'][:8]}): ordinal {missing} "
            f"never arrived in {stalled_for:.0f}s; sealed {sealed}, failed {failed}"
        )
        return event

    # ------------------------------ claimed TTL ------------------------------

    def claimed_timeouts(self, now: float | None = None) -> list[dict]:
        """Over-age CLAIMED operations for the backend to terminal-fail (an orphaned claim blocks its queue forever)."""
        now = self._time() if now is None else now
        if self.claimed_ttl is None or self.claimed_ttl <= 0:
            return []
        return [
            {**op.view(), "claimed_age": now - op.claimed_at}
            for op in self.by_id.values()
            if op.state is OperationState.CLAIMED
            and op.claimed_at is not None
            and now - op.claimed_at >= self.claimed_ttl
        ]

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

    def mark_window_consumed(self, operation_id: str) -> None:
        op = self.by_id.get(operation_id)
        if op is not None:
            op.window_consumed = True

    def cancel(self, operation_id: str) -> dict:
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

    def payload(self, operation_id: str) -> dict | None:
        """The stored request payload (metrics recomputation at completion)."""
        op = self.by_id.get(operation_id)
        return op.payload if op is not None else None

    def ack(self, operation_id: str) -> None:
        op = self.by_id.get(operation_id)
        if op is None:
            return
        if not op.terminal:
            raise ValueError(f"operation '{operation_id}' is {op.state.value}; ack applies to terminal operations")
        self.by_id.pop(operation_id, None)
        # The ordinal slot stays reserved (contiguity/uniqueness), but an acked
        # record's payload and result are released — they can be large.
        op.payload = {}
        op.result = None
        queue = self.queues.get(op.tenant)
        if queue is not None:
            queue.operations = [o for o in queue.operations if o.operation_id != operation_id]
            # by_ordinal keeps the slot so contiguity and ordinal uniqueness survive the ack.
            if not queue.operations and queue.fenced:
                self.queues.pop(op.tenant, None)

    # ------------------------------ fencing ------------------------------

    def fence(self, name: str, registration_id: str) -> list[str]:
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
