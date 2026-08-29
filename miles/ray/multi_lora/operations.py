"""Client-driven operation queue: per registration, one ordinal-keyed dict serves as reorder buffer, idempotency table, and result store."""

import hashlib
import json
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass, field

DATA_KINDS = ("forward_backward", "forward")
CONTROL_KINDS = ("optim_step", "save_weights_for_sampler", "save_state", "load_state")


class QueueFull(RuntimeError):
    """Capacity reached; the wire layer maps this to 429 (sampling plane only, never training)."""

    def __init__(self, client_detail: str):
        super().__init__(client_detail)
        self.client_detail = client_detail  # authored contract text, safe for external responses


class BadRequest(ValueError):
    """Contract violation by the client; ValueError so the control-plane handler maps it to 400."""

    def __init__(self, client_detail: str):
        super().__init__(client_detail)
        self.client_detail = client_detail  # authored contract text, safe for external responses


def payload_fingerprint(kind: str, payload: dict) -> str:
    """Digest of identity-relevant content; callers must exclude volatile per-retry fields first."""
    canonical = json.dumps({"k": kind, "p": payload or {}}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class OperationRecord:
    ordinal: int
    request_id: str
    kind: str
    payload: dict
    fingerprint: str
    status: str = "QUEUED"  # QUEUED|RUNNING|DONE|FAILED
    result: dict | None = None
    error: str | None = None
    error_kind: str | None = None  # user|server
    delivered: bool = False
    evicted: bool = False


@dataclass
class OperationQueue:
    """Per-registration ordinal queue; cap=None (training plane) because capacity-429 deadlocks the SDK, which posts chunk 1 last."""

    cap: int | None = None
    keep_delivered: int = 4
    ops: dict[int, OperationRecord] = field(default_factory=dict)
    by_request_id: dict[str, int] = field(default_factory=dict)
    next_to_run: int = 1
    poisoned: bool = False  # a forward_backward FAILED since the last optim_step terminal
    fenced: bool = False  # retired registration: replays still answer, new ops are rejected

    def __post_init__(self):
        # Retained delivered terminals consume capacity: keep_delivered >= cap wedges the queue permanently.
        if self.cap is not None and self.keep_delivered >= self.cap:
            raise ValueError("keep_delivered must be < cap (retained terminals consume capacity)")

    # ------------------------------ enqueue ------------------------------

    def enqueue(
        self, ordinal: int, request_id: str, kind: str, payload: dict | None = None, *, fingerprint: str | None = None
    ) -> int:
        """Buffer one operation; retry identity is `fingerprint` (defaults to hashing `payload`), execution input is `payload`."""
        fp = fingerprint if fingerprint is not None else payload_fingerprint(kind, payload)
        if (known := self.by_request_id.get(request_id)) is not None:
            rec = self.ops[known]
            if rec.fingerprint != fp or rec.ordinal != ordinal:
                raise BadRequest(f"request '{request_id}' retried with different content; retries must be identical")
            return known
        if self.fenced:
            raise BadRequest("registration retired; new operations are fenced")
        if ordinal in self.ops:
            raise BadRequest(f"ordinal {ordinal} already taken by request '{self.ops[ordinal].request_id}'")
        if ordinal < self.next_to_run:
            raise BadRequest(f"ordinal {ordinal} is in the executed past (cursor at {self.next_to_run})")
        if self.cap is not None and self._live_count() >= self.cap and not self._fills_blocking_gap(ordinal):
            raise QueueFull(f"{self._live_count()} operations held (cap {self.cap})")
        self.ops[ordinal] = OperationRecord(ordinal, request_id, kind, dict(payload or {}), fp)
        self.by_request_id[request_id] = ordinal
        return ordinal

    def _live_count(self) -> int:
        return sum(1 for r in self.ops.values() if not r.evicted)

    def _fills_blocking_gap(self, ordinal: int) -> bool:
        """Admit past cap only the ordinal that unblocks queued ops above it, hard-ceilinged at 2*cap."""
        if not self.ops:
            return False
        if self._live_count() >= 2 * self.cap:
            return False
        return ordinal == self._first_missing() and ordinal < max(self.ops)

    def _first_missing(self) -> int:
        ordinal = self.next_to_run
        while ordinal in self.ops:
            ordinal += 1
        return ordinal

    def open_count(self) -> int:
        return sum(1 for rec in self.ops.values() if rec.status in ("QUEUED", "RUNNING"))

    # ------------------------------ poll / eviction ------------------------------

    def poll(self, ordinal: int) -> tuple[str, dict | None]:
        """Answer ("result", payload) | ("error", {error, category}) | ("try_again", None)."""
        rec = self.ops.get(ordinal)
        if rec is None:
            raise BadRequest(f"unknown ordinal {ordinal}")
        if rec.evicted:
            return ("error", {"error": "result expired (already delivered once)", "category": "user"})
        if rec.status == "DONE":
            rec.delivered = True
            # Delivery order is not ordinal order: capture the answer BEFORE eviction can null it.
            result = rec.result
            self._evict()
            return ("result", result)
        if rec.status == "FAILED":
            rec.delivered = True
            error = {"error": rec.error, "category": rec.error_kind or "server"}
            self._evict()
            return ("error", error)
        return ("try_again", None)

    def _evict(self) -> None:
        """Strip delivered terminals beyond keep_delivered; tombstones keep ordinal + request-id identity."""
        delivered = sorted(
            (r for r in self.ops.values() if r.delivered and not r.evicted and r.status in ("DONE", "FAILED")),
            key=lambda r: r.ordinal,
        )
        while len(delivered) > self.keep_delivered:
            victim = delivered.pop(0)
            victim.payload = {}
            victim.result = None
            victim.evicted = True

    # ------------------------------ collect / terminals ------------------------------

    def claim_next_runnable_ops(self) -> list[OperationRecord]:
        """Claim (mark RUNNING) the next runnable ops at the cursor: consecutive data ops, or one control op alone."""
        out: list[OperationRecord] = []
        ordinal = self.next_to_run
        while (rec := self.ops.get(ordinal)) is not None and rec.status == "QUEUED":
            if rec.kind in CONTROL_KINDS:
                if out:
                    break
                if rec.kind == "optim_step" and self.poisoned:
                    # A failed forward_backward left partial grads: stepping would corrupt the adapter.
                    self._fail_inline(
                        rec, "a forward_backward in this optimizer window failed; redo the cycle", "user"
                    )
                    ordinal += 1
                    continue
                rec.status = "RUNNING"
                return [rec]
            rec.status = "RUNNING"
            out.append(rec)
            ordinal += 1
        return out

    def fence(self, reason: str) -> None:
        """Retire the queue: every non-terminal op fails as a user error; replays keep answering."""
        self.fenced = True
        for rec in list(self.ops.values()):
            if rec.status in ("QUEUED", "RUNNING"):
                self.fail(rec.ordinal, reason, "user")

    def complete(self, ordinal: int, result: dict | None = None) -> None:
        rec = self.ops[ordinal]
        if rec.status in ("DONE", "FAILED"):  # fence-vs-driver races resolve to the first terminal
            return
        rec.status = "DONE"
        rec.result = result
        if rec.kind == "optim_step":
            self.poisoned = False
        self.next_to_run = max(self.next_to_run, ordinal + 1)

    def fail(self, ordinal: int, error: str, kind: str = "server") -> None:
        rec = self.ops[ordinal]
        if rec.status in ("DONE", "FAILED"):
            return
        rec.status = "FAILED"
        rec.error = error
        rec.error_kind = kind
        if rec.kind in DATA_KINDS:
            self.poisoned = True
        elif rec.kind == "optim_step":
            self.poisoned = False
        self.next_to_run = max(self.next_to_run, ordinal + 1)

    def _fail_inline(self, rec: OperationRecord, error: str, kind: str) -> None:
        rec.status = "FAILED"
        rec.error = error
        rec.error_kind = kind
        if rec.kind == "optim_step":
            self.poisoned = False  # any optim terminal closes the window
        self.next_to_run = max(self.next_to_run, rec.ordinal + 1)


class OperationQueueSet:
    """Rotation-fair container of per-registration queues; eligibility is the caller's knowledge."""

    def __init__(self) -> None:
        self.queues: dict[str, OperationQueue] = {}
        self._rotation: deque[str] = deque()

    def get_or_create(self, name: str) -> OperationQueue:
        if name not in self.queues:
            self.queues[name] = OperationQueue()
            self._rotation.append(name)
        return self.queues[name]

    def replace(self, name: str) -> OperationQueue:
        """Fresh queue for a new registration life; the predecessor drops with its tombstones."""
        self.queues[name] = OperationQueue()
        if name not in self._rotation:
            self._rotation.append(name)
        return self.queues[name]

    def fence(self, name: str, reason: str) -> None:
        queue = self.queues.get(name)
        if queue is not None:
            queue.fence(reason)

    def rotation_pass(self) -> list[str]:
        """Snapshot in rotation order (pruning names whose queue is gone), then advance the head."""
        live = [name for name in self._rotation if name in self.queues]
        self._rotation = deque(live)
        if self._rotation:
            self._rotation.rotate(-1)
        return live

    def claim_rounds(self, eligible: Collection[str]) -> list[tuple[str, list[OperationRecord]]]:
        """One claim per eligible queue in rotation order; empty claims drop out."""
        rounds = []
        for name in self.rotation_pass():
            if name not in eligible:
                continue
            claimed = self.queues[name].claim_next_runnable_ops()
            if claimed:
                rounds.append((name, claimed))
        return rounds

    def complete(self, results: list[dict]) -> None:
        """Apply executor outcomes; first-terminal-wins in each queue makes fence races safe."""
        for outcome in results:
            queue = self.queues.get(outcome["name"])
            if queue is None:
                continue
            if outcome.get("ok", False):
                queue.complete(outcome["ordinal"], outcome.get("result"))
            else:
                queue.fail(
                    outcome["ordinal"], outcome.get("error", "operation failed"), outcome.get("category", "server")
                )

    def depths(self) -> dict[str, int]:
        return {name: queue.open_count() for name, queue in self.queues.items()}
