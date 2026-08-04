from typing import Any

import pytest
from pydantic import ValidationError

from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort


class _FakeWorkerHandle(BaseWorkerHandle):
    async def wait_ready(self, *, timeout: float) -> None:
        return None

    async def wait_dead(self, *, timeout: float) -> None:
        return None


def _fields(**overrides: Any) -> dict[str, Any]:
    return (
        dict(
            name="engine-0-0",
            generation=1,
            self_addrs={"primary": HostAndPort(host="10.0.0.1", port=30000)},
            gpu_ids=[0],
            handle=_FakeWorkerHandle(),
        )
        | overrides
    )


class TestWorkerInfoValidation:
    def test_an_unknown_field_is_rejected(self):
        """A misspelled field must fail loudly instead of silently dropping out of the description."""
        with pytest.raises(ValidationError, match="gpu_id"):
            WorkerInfo(**_fields(gpu_id=[0]))

    def test_a_non_worker_handle_is_rejected(self):
        """The handle is the only way to reach the worker, so a raw actor or stray object must not pass."""
        with pytest.raises(ValidationError, match="BaseWorkerHandle"):
            WorkerInfo(**_fields(handle=object()))
