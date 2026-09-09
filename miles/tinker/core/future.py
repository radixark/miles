"""Future store: the ledger behind submit-then-poll.

The HTTP side reads, the execution side writes; this is the only interface
between the two worlds. Futures live in memory: a gateway restart answers
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
class Future:
    request_id: str
    model_id: str
    tenant: str
    state: str = PENDING
    result: dict | None = None
    error: str | None = None
    error_category: str | None = None
    finished_at: float | None = None
    created_at: float = field(default_factory=time.monotonic)


class FutureStore:
    def __init__(self) -> None:
        self._futures: dict[str, Future] = {}

    def create(self, model_id: str, tenant: str) -> Future:
        future = Future(request_id=f"req-{uuid.uuid4().hex}", model_id=model_id, tenant=tenant)
        self._futures[future.request_id] = future
        return future

    def resolve(self, request_id: str, result: dict) -> None:
        future = self._futures[request_id]
        future.state = DONE
        future.result = result
        future.finished_at = time.monotonic()

    def fail(self, request_id: str, error: str, category: str) -> None:
        future = self._futures[request_id]
        future.state = FAILED
        future.error = error
        future.error_category = category
        future.finished_at = time.monotonic()

    def get(self, request_id: str, tenant: str) -> Future | None:
        """None means unknown/expired: the HTTP layer answers 410 and the SDK
        resubmits the original request."""
        self._sweep()
        future = self._futures.get(request_id)
        if future is None:
            return None
        if future.tenant != tenant:
            raise OwnershipError(f"request {request_id} does not belong to this tenant")
        return future

    def _sweep(self) -> None:
        now = time.monotonic()
        expired = [
            rid
            for rid, p in self._futures.items()
            if p.finished_at is not None and now - p.finished_at > _FINISHED_TTL_S
        ]
        for rid in expired:
            del self._futures[rid]
