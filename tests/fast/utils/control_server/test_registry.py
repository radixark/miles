from __future__ import annotations

import pytest

from miles.utils.control_server.registry import _CellRegistry

from .conftest import MockHandle


class TestCellRegistry:
    def test_register_and_get_by_id(self, registry: _CellRegistry) -> None:
        handle = MockHandle(cell_id="cell-0", cell_type="rollout")
        registry.register(handle)
        assert registry.get("cell-0") is handle

    def test_get_unknown_id_raises_key_error(self, registry: _CellRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_all_returns_all_registered(self, registry: _CellRegistry) -> None:
        h1 = MockHandle(cell_id="cell-0", cell_type="rollout")
        h2 = MockHandle(cell_id="cell-1", cell_type="rollout")
        registry.register(h1)
        registry.register(h2)

        all_handles = registry.get_all()
        assert len(all_handles) == 2
        assert h1 in all_handles
        assert h2 in all_handles

    def test_register_duplicate_id_raises(self, registry: _CellRegistry) -> None:
        h1 = MockHandle(cell_id="cell-0", cell_type="rollout")
        h2 = MockHandle(cell_id="cell-0", cell_type="rollout")
        registry.register(h1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(h2)
