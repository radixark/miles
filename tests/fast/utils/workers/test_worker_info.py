from typing import Any

import pytest
from pydantic import ValidationError

from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort


def _fields(**overrides: Any) -> dict[str, Any]:
    return (
        dict(
            name="engine-0-0",
            generation=1,
            self_addrs={"primary": HostAndPort(host="10.0.0.1", port=30000)},
            gpu_ids=[0],
        )
        | overrides
    )


class TestWorkerInfoValidation:
    def test_an_unknown_field_is_rejected(self):
        """A misspelled field must fail loudly instead of silently dropping out of the description."""
        with pytest.raises(ValidationError, match="gpu_id"):
            WorkerInfo(**_fields(gpu_id=[0]))

    def test_a_non_string_worker_class_is_rejected(self):
        """The class crosses the wire as an import path, so a live object must not pass for it."""
        with pytest.raises(ValidationError, match="worker_class"):
            WorkerInfo(**_fields(worker_class=object()))
