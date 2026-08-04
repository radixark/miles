# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

ObjectKey = str
ParentKey = str


@dataclass(frozen=True)
class UpsertEvent:
    key: ObjectKey
    obj: Any


@dataclass(frozen=True)
class DeleteEvent:
    key: ObjectKey
    last_obj: Any


SourceEvent = UpsertEvent | DeleteEvent

SourceWatchFn = Callable[[], AsyncGenerator[SourceEvent, None]]
