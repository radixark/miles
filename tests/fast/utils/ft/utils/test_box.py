"""Unit tests for Box[T] (P0 item 9)."""

from __future__ import annotations

from miles.utils.ft.utils.box import Box


class TestBox:
    def test_store_and_retrieve_value(self) -> None:
        box: Box[int] = Box(42)
        assert box.value == 42

    def test_mutate_value(self) -> None:
        box: Box[str | None] = Box(None)
        assert box.value is None

        box.value = "hello"
        assert box.value == "hello"

    def test_shared_reference(self) -> None:
        """Two references to the same Box see the same mutation."""
        box: Box[int] = Box(0)
        ref = box

        box.value = 99
        assert ref.value == 99
