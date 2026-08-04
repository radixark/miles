# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.reconcile.source_event import DeleteEvent, ObjectKey, ParentKey, SourceEvent, UpsertEvent

logger = logging.getLogger(__name__)

KeyMapFn = Callable[[Any], ParentKey | None]


@dataclass(frozen=True)
class _CachedObject:
    obj: Any
    parent: ParentKey


class ObjectStore:
    def __init__(self, *, key_map: KeyMapFn | None) -> None:
        self._key_map = key_map
        self._cache: dict[ObjectKey, _CachedObject] = {}

    def get_by_parent(self, parent_key: ParentKey) -> list[Any]:
        return [self._cache[key].obj for key in sorted(self._cache) if self._cache[key].parent == parent_key]

    def __contains__(self, key: ObjectKey) -> bool:
        return key in self._cache

    def handle_event(self, event: SourceEvent) -> set[ParentKey]:
        match event:
            case UpsertEvent():
                return self._handle_upsert(event)
            case DeleteEvent():
                return self._handle_delete(event)
            case _:
                raise AssertionError(f"Unknown source event {event=}")

    def _handle_upsert(self, event: UpsertEvent) -> set[ParentKey]:
        parent = self._parent_key_or_none(key=event.key, obj=event.obj)
        if parent is not None:
            return self._apply_upsert(key=event.key, obj=event.obj, parent=parent)
        if event.key not in self._cache:
            return set()
        return self._apply_delete(key=event.key, last_obj=None)

    def _handle_delete(self, event: DeleteEvent) -> set[ParentKey]:
        return self._apply_delete(key=event.key, last_obj=event.last_obj)

    def _apply_upsert(self, *, key: ObjectKey, obj: Any, parent: ParentKey) -> set[ParentKey]:
        previous = self._cache.get(key)
        self._cache[key] = _CachedObject(obj=obj, parent=parent)
        return {parent} if previous is None else {parent, previous.parent}

    def _apply_delete(self, *, key: ObjectKey, last_obj: Any) -> set[ParentKey]:
        removed = self._cache.pop(key, None)
        parent = removed.parent if removed is not None else None
        if parent is None:
            if last_obj is None:
                logger.warning(f"ObjectStore dropping a delete it cannot attribute to a parent {key=}")
                return set()
            parent = self._parent_key_or_none(key=key, obj=last_obj)
            if parent is None:
                return set()
        return {parent}

    def _parent_key_or_none(self, *, key: ObjectKey, obj: Any) -> ParentKey | None:
        try:
            return key if self._key_map is None else self._key_map(obj)
        except Exception:
            logger.error(f"ObjectStore dropping an object whose key_map failed {key=}", exc_info=True)
            return None
