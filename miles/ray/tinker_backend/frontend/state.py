"""Frontend-owned protocol state: sessions, models, futures, checkpoints,
sampling sessions.

Identity is deterministic wherever the SDK retries: the SDK addresses work
by (session_id, model_seq_id) and (model, seq_id), so request ids derive
from those coordinates and a resent submission finds its original record.
Every record carries the fingerprint of the request that minted it — an
identical retry replays, a different payload at the same coordinates is a
conflict (422; the SDK treats 409 as retryable, so a true conflict must
never be a 409).

All state is in-memory and single-writer: the frontend runs on the
controller actor's event loop, and store mutations never straddle an await.
Terminal future bodies are kept for replay (a response lost on the wire is
re-polled) inside a bounded LRU of delivered results; eviction keeps a
compact fingerprint tombstone so an expired identity answers a typed 410
instead of silently re-executing.
"""

import hashlib
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


def fingerprint_of(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ConflictError(ValueError):
    """Same identity, different content: the client must not silently retry."""


class ExpiredError(ValueError):
    """The result was delivered and its replay window has expired. The exact
    terminal bytes are gone, so neither replay nor re-execution is possible —
    re-running would break idempotency (a fresh sample for a spent seq, an
    ordinal the ledger already consumed). Maps to a typed 410."""


def _check_fingerprint(kind: str, key: str, existing: str, incoming: str) -> None:
    if existing != incoming:
        raise ConflictError(f"{kind} '{key}' already exists with a different request; retries must be identical")


@dataclass
class SessionRecord:
    session_id: str
    sdk_version: str = ""
    tags: list[str] = field(default_factory=list)
    user_metadata: dict | None = None
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def short(self) -> str:
        return self.session_id.removeprefix("sess-")[:12]


class SessionStore:
    def __init__(self) -> None:
        self.records: dict[str, SessionRecord] = {}

    def create(self, sdk_version: str, tags: list[str], user_metadata: dict | None) -> SessionRecord:
        record = SessionRecord(
            session_id=f"sess-{uuid.uuid4().hex[:16]}",
            sdk_version=sdk_version,
            tags=tags,
            user_metadata=user_metadata,
        )
        self.records[record.session_id] = record
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        return self.records.get(session_id)

    def heartbeat(self, session_id: str) -> bool:
        record = self.records.get(session_id)
        if record is None:
            return False
        record.last_heartbeat = time.time()
        return True


@dataclass
class ModelRecord:
    """One SDK training client == one backend registration."""

    model_id: str  # public: "{session_id}:train:{model_seq_id}" (official shape)
    session_id: str
    model_seq_id: int
    name: str  # backend adapter name
    registration_id: str
    base_model: str
    rank: int
    fingerprint: str

    @property
    def rid8(self) -> str:
        return self.registration_id[:8]


class ModelStore:
    def __init__(self) -> None:
        self.by_model_id: dict[str, ModelRecord] = {}

    def add(self, record: ModelRecord) -> None:
        self.by_model_id[record.model_id] = record

    def get(self, model_id: str) -> ModelRecord | None:
        return self.by_model_id.get(model_id)


@dataclass
class FutureRecord:
    """One retrievable request_id. ``terminal`` holds the exact JSON body to
    replay once resolved; until then ``kind`` picks the resolver."""

    request_id: str
    kind: str  # "operation" | "create_model" | "unload_model" | "sample"
    fingerprint: str
    model: ModelRecord | None = None
    operation_id: str | None = None
    operation_kind: str | None = None
    # forward results need a metrics recompute from the request payload (the
    # backend attaches metrics to forward_backward only); dropped when terminal.
    forward_payload: dict | None = None
    # save/load bookkeeping minted at submit time.
    tinker_path: str | None = None
    backend_target: dict | None = None
    # ephemeral publish: the sampling session to mint at completion.
    sampling_session_id: str | None = None
    terminal: dict | None = None
    created_at: float = field(default_factory=time.time)

    def resolve(self, body: dict) -> dict:
        self.terminal = body
        self.forward_payload = None
        return body


class FutureStore:
    """request_id -> FutureRecord with bounded retention of delivered
    terminal results (replay window for lost responses). Eviction leaves a
    compact tombstone (request_id -> fingerprint): the record's identity
    outlives its bytes, so a late identical retry gets a truthful typed 410
    instead of silently re-executing (samples would re-generate, training
    ordinals would collide with the ledger) or a misleading conflict."""

    def __init__(self, max_delivered: int = 4096, max_expired: int = 65536) -> None:
        self.records: dict[str, FutureRecord] = {}
        self.max_delivered = max_delivered
        self.max_expired = max_expired
        self._delivered: OrderedDict[str, None] = OrderedDict()
        self._expired: OrderedDict[str, str] = OrderedDict()

    def put(self, record: FutureRecord) -> FutureRecord:
        self.records[record.request_id] = record
        return record

    def get(self, request_id: str) -> FutureRecord | None:
        return self.records.get(request_id)

    def expired_fingerprint(self, request_id: str) -> str | None:
        return self._expired.get(request_id)

    def existing(self, request_id: str, fingerprint: str) -> FutureRecord | None:
        """The idempotent-retry lookup: same id + same fingerprint replays,
        same id + different content conflicts, delivered-then-evicted expires."""
        record = self.records.get(request_id)
        if record is None:
            expired = self._expired.get(request_id)
            if expired is not None:
                _check_fingerprint("request", request_id, expired, fingerprint)
                raise ExpiredError(
                    f"request '{request_id}' was already delivered and its replay window expired; "
                    "the original result cannot be reproduced"
                )
            return None
        _check_fingerprint("request", request_id, record.fingerprint, fingerprint)
        return record

    def mark_delivered(self, record: FutureRecord) -> None:
        if record.terminal is None:
            return
        self._delivered[record.request_id] = None
        self._delivered.move_to_end(record.request_id)
        while len(self._delivered) > self.max_delivered:
            evicted, _ = self._delivered.popitem(last=False)
            dropped = self.records.pop(evicted, None)
            if dropped is not None:
                self._expired[evicted] = dropped.fingerprint
        while len(self._expired) > self.max_expired:
            self._expired.popitem(last=False)


@dataclass
class CheckpointRecord:
    tinker_path: str  # public "tinker://{run}/weights/{tag}"
    backend_path: str  # trainer-side state directory
    name: str
    registration_id: str
    base_model: str
    rank: int
    step: int


class CheckpointCatalog:
    """tinker:// URI -> backend state path. In-memory: paths minted by this
    controller lifetime resolve; the artifacts themselves persist on disk."""

    def __init__(self) -> None:
        self.records: dict[str, CheckpointRecord] = {}

    def add(self, record: CheckpointRecord) -> None:
        self.records[record.tinker_path] = record

    def get(self, tinker_path: str) -> CheckpointRecord | None:
        return self.records.get(tinker_path)


@dataclass
class SamplingSessionRecord:
    sampling_session_id: str
    session_id: str
    fingerprint: str
    base_model: str
    # None for base-model sessions; set for ephemeral LoRA publishes.
    name: str | None = None
    registration_id: str | None = None
    serving_name: str | None = None
    serving_version: int | None = None
    # Compact spent-sequence fence: sample identities outlive the bounded
    # future/tombstone retention. Every seq <= spent_fence has executed;
    # spent_sparse holds executed seqs above the fence (out-of-order arrival
    # gaps only, so it stays tiny for the SDK's monotonic counters). A retry
    # of a spent seq whose bytes and tombstone are both gone gets a typed
    # terminal failure instead of silently re-running the generation.
    spent_fence: int = -1
    spent_sparse: set = field(default_factory=set)

    def is_spent(self, seq_id: int) -> bool:
        return seq_id <= self.spent_fence or seq_id in self.spent_sparse

    def mark_spent(self, seq_id: int) -> None:
        if self.is_spent(seq_id):
            return
        self.spent_sparse.add(seq_id)
        while self.spent_fence + 1 in self.spent_sparse:
            self.spent_fence += 1
            self.spent_sparse.discard(self.spent_fence)


class SamplingSessionStore:
    def __init__(self) -> None:
        self.records: dict[str, SamplingSessionRecord] = {}

    def add(self, record: SamplingSessionRecord) -> SamplingSessionRecord:
        self.records[record.sampling_session_id] = record
        return record

    def get(self, sampling_session_id: str) -> SamplingSessionRecord | None:
        return self.records.get(sampling_session_id)

    def existing(self, sampling_session_id: str, fingerprint: str) -> SamplingSessionRecord | None:
        record = self.records.get(sampling_session_id)
        if record is None:
            return None
        _check_fingerprint("sampling session", sampling_session_id, record.fingerprint, fingerprint)
        return record
