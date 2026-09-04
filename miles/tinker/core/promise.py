"""Promise store: the ledger behind submit-then-poll.

The HTTP side reads, the execution side writes; this is the only interface
between the two worlds. Promises live in memory: a gateway restart answers
410 and the SDK resubmits the original request, which the stream dedup makes
safe.
"""

import time
import uuid
from dataclasses import dataclass, field

from miles.tinker.core.types import OwnershipError

PENDING = "pending"
DONE = "done"
FAILED = "failed"

_FINISHED_TTL_S = 3600.0


@dataclass
class Promise:
    request_id: str
    model_id: str
    tenant: str
    state: str = PENDING
    result: dict | None = None
    error: str | None = None
    error_category: str | None = None
    finished_at: float | None = None
    created_at: float = field(default_factory=time.monotonic)


class PromiseStore:
    def __init__(self) -> None:
        self._promises: dict[str, Promise] = {}

    def create(self, model_id: str, tenant: str) -> Promise:
        promise = Promise(request_id=f"req-{uuid.uuid4().hex}", model_id=model_id, tenant=tenant)
        self._promises[promise.request_id] = promise
        return promise

    def resolve(self, request_id: str, result: dict) -> None:
        promise = self._promises[request_id]
        promise.state = DONE
        promise.result = result
        promise.finished_at = time.monotonic()

    def fail(self, request_id: str, error: str, category: str) -> None:
        promise = self._promises[request_id]
        promise.state = FAILED
        promise.error = error
        promise.error_category = category
        promise.finished_at = time.monotonic()

    def get(self, request_id: str, tenant: str) -> Promise | None:
        """None means unknown/expired: the HTTP layer answers 410 and the SDK
        resubmits the original request."""
        self._sweep()
        promise = self._promises.get(request_id)
        if promise is None:
            return None
        if promise.tenant != tenant:
            raise OwnershipError(f"request {request_id} does not belong to this tenant")
        return promise

    def _sweep(self) -> None:
        now = time.monotonic()
        expired = [
            rid
            for rid, p in self._promises.items()
            if p.finished_at is not None and now - p.finished_at > _FINISHED_TTL_S
        ]
        for rid in expired:
            del self._promises[rid]
