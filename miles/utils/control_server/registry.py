from __future__ import annotations

from miles.utils.control_server.handles import _CellHandle


class _CellRegistry:
    def __init__(self) -> None:
        self._handles: dict[str, _CellHandle] = {}

    def register(self, handle: _CellHandle) -> None:
        if handle.cell_id in self._handles:
            raise ValueError(f"Cell '{handle.cell_id}' is already registered")
        self._handles[handle.cell_id] = handle

    def get_all(self) -> list[_CellHandle]:
        return list(self._handles.values())

    def get(self, cell_id: str) -> _CellHandle:
        return self._handles[cell_id]
