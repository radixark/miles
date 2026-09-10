"""In-memory future results. Request deduplication lasts only for the current process."""

import time
import uuid
from dataclasses import dataclass

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


class FutureStore:
    def __init__(self) -> None:
        self._futures: dict[str, Future] = {}

    def create(self, model_id: str, tenant: str) -> Future:
        future = Future(request_id=f"req-{uuid.uuid4().hex}", model_id=model_id, tenant=tenant)
        self._futures[future.request_id] = future
        return future

    def resolve(self, request_id: str, result: dict) -> None:
        future = self._futures[request_id]
        if future.state != PENDING:  # e.g. already failed by lease expiry
            return
        future.state = DONE
        future.result = result
        future.finished_at = time.monotonic()

    def fail(self, request_id: str, error: str, category: str) -> None:
        future = self._futures[request_id]
        if future.state != PENDING:
            return
        future.state = FAILED
        future.error = error
        future.error_category = category
        future.finished_at = time.monotonic()

    def get(self, request_id: str, tenant: str) -> Future | None:
        """Return None for unknown or expired futures; enforce tenant ownership otherwise."""
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
            request_id
            for request_id, future in self._futures.items()
            if future.finished_at is not None and now - future.finished_at > _FINISHED_TTL_S
        ]
        for request_id in expired:
            del self._futures[request_id]
