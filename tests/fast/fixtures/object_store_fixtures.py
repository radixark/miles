from __future__ import annotations

import copy
import itertools
from typing import Any

import pytest

from miles.utils import object_store
from miles.utils.object_store import (
    BaseObjectStore,
    ObjectStoreGetResult,
    StoreObjectRef,
    ValueSpec,
    _MooncakeStoreObjectRef,
)


class InMemoryStore(BaseObjectStore):
    def __init__(self) -> None:
        self.objects: dict[int, Any] = {}
        self.removed: list[int] = []
        self._keys = itertools.count()

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        key = next(self._keys)
        self.objects[key] = copy.deepcopy(value)
        return _MooncakeStoreObjectRef(payload={"key": key})

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        return ObjectStoreGetResult(value=self.objects[ref.payload["key"]], release_fn=lambda value: None)

    def remove(self, ref: StoreObjectRef) -> None:
        del self.objects[ref.payload["key"]]
        self.removed.append(ref.payload["key"])


@pytest.fixture
def in_memory_store(monkeypatch: pytest.MonkeyPatch) -> InMemoryStore:
    instance = InMemoryStore()
    monkeypatch.setattr(object_store, "_INSTANCE", instance)
    return instance
