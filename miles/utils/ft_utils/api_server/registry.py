from __future__ import annotations

import asyncio

from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import Cell


class _CellRegistry:
    def __init__(self, handlers: list[_CellHandler]) -> None:
        cell_types = [handler.cell_type for handler in handlers]
        assert len(set(cell_types)) == len(cell_types), f"Duplicate cell types: {cell_types}"
        self._handlers = handlers

    async def list_cells(self) -> list[Cell]:
        per_handler = await asyncio.gather(*(handler.list_cells() for handler in self._handlers))
        return [cell for cells in per_handler for cell in cells]

    async def resolve(self, cell_id: str) -> _CellHandler:
        for handler in self._handlers:
            if cell_id in await handler.list_cell_ids():
                return handler
        raise KeyError(cell_id)
