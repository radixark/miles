from __future__ import annotations

from typing import Any

import pytest

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.utils import object_store
from miles.utils.data import remove_train_output_refs
from miles.utils.object_store import BaseObjectStore, ObjectStoreGetResult, StoreObjectRef, ValueSpec


def _ref(payload: Any) -> StoreObjectRef:
    return StoreObjectRef(payload=payload)


class _RecordingStore(BaseObjectStore):
    def __init__(self) -> None:
        self.removed: list[StoreObjectRef] = []

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        return _ref(value)

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        raise NotImplementedError

    def remove(self, ref: StoreObjectRef) -> None:
        self.removed.append(ref)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    instance = _RecordingStore()
    monkeypatch.setattr(object_store, "_INSTANCE", instance)
    return instance


class TestRemoveTrainOutputRefs:
    def test_every_shipped_ref_is_released(self, store: _RecordingStore):
        """Nothing else frees these objects, so a missed ref leaks for the whole run under mooncake."""
        refs = [_ref("a"), _ref("b")]

        remove_train_output_refs([TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref) for ref in refs])

        assert store.removed == refs

    def test_a_worker_that_shipped_nothing_is_skipped(self, store: _RecordingStore):
        """Only pp-last-stage critic workers ship values, so the rest carry None and must not reach the store."""
        ref = _ref("a")

        remove_train_output_refs(
            [
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None),
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref),
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None),
            ]
        )

        assert store.removed == [ref]
