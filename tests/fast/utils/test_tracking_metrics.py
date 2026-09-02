"""jsonable_metrics: metric values pydantic cannot serialise must be unwrapped."""

import numpy as np
import pytest
import torch

from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.tracking_utils.tracking import jsonable_metrics


def _event(metrics: dict) -> MetricEvent:
    from datetime import datetime, timezone

    return MetricEvent(metrics=metrics, timestamp=datetime.now(timezone.utc), source=MainProcessIdentity())


class TestJsonableMetrics:
    def test_unwraps_numpy_scalars(self) -> None:
        out = jsonable_metrics({"i": np.int64(3), "f": np.float32(1.5), "b": np.bool_(True)})
        assert out == {"i": 3, "f": pytest.approx(1.5), "b": True}
        assert all(not isinstance(v, np.generic) for v in out.values())

    def test_unwraps_torch_scalars(self) -> None:
        assert jsonable_metrics({"t": torch.tensor(2.0)}) == {"t": pytest.approx(2.0)}

    def test_passes_everything_else_through_untouched(self) -> None:
        metrics = {"i": 1, "f": 0.5, "s": "step", "n": None, "d": {"a": 1}, "l": [1, 2]}
        assert jsonable_metrics(metrics) == metrics

    def test_numpy_scalar_metric_is_serialisable_after_unwrapping(self) -> None:
        """The regression: a numpy scalar reaching model_dump_json() raised
        PydanticSerializationError and killed the training step. Only runs with the
        event logger initialised (--dump-details / --save-debug-event-data) hit it."""
        raw = {"rollout/turns": np.int64(7)}
        with pytest.raises(Exception):
            _event(raw).model_dump_json()
        assert _event(jsonable_metrics(raw)).model_dump_json()
