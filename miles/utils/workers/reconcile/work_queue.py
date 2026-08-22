# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Hashable
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT", bound=Hashable)


class WorkQueue(Generic[KeyT]):
    def __init__(self) -> None:
        self._keys: deque[KeyT] = deque()
        self._wakeup = asyncio.Event()
        self._shutdown = False

    def add(self, key: KeyT) -> None:
        if self._shutdown:
            return
        if key not in self._keys:
            self._keys.append(key)
        self._wakeup.set()

    async def get(self) -> KeyT | None:
        while not self._shutdown:
            if self._keys:
                return self._keys.popleft()
            self._wakeup.clear()
            await self._wakeup.wait()
        return None

    def shutdown(self) -> None:
        self._shutdown = True
        self._wakeup.set()
