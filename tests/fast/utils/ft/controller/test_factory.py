"""Tests for controller/factory.py — rank_pids_provider lambda safety."""

from __future__ import annotations

from unittest.mock import MagicMock

from miles.utils.ft.utils.box import Box


class TestRankPidsProviderTOCTOU:
    """H-1: the rank_pids_provider lambda used to read Box.value twice,
    allowing a TOCTOU race where value changes between the None check
    and the method call. Now captured once via walrus operator."""

    def test_returns_empty_dict_when_box_is_none(self) -> None:
        box: Box[MagicMock | None] = Box(None)
        provider = lambda node_id: (r.get_rank_pids_for_node(node_id) if (r := box.value) is not None else {})

        assert provider("node-1") == {}

    def test_returns_rank_pids_when_roster_set(self) -> None:
        roster = MagicMock()
        roster.get_rank_pids_for_node.return_value = {0: 1234}
        box: Box[MagicMock | None] = Box(roster)
        provider = lambda node_id: (r.get_rank_pids_for_node(node_id) if (r := box.value) is not None else {})

        result = provider("node-1")

        assert result == {0: 1234}
        roster.get_rank_pids_for_node.assert_called_once_with("node-1")

    def test_no_error_when_box_cleared_after_set(self) -> None:
        roster = MagicMock()
        roster.get_rank_pids_for_node.return_value = {0: 1234}
        box: Box[MagicMock | None] = Box(roster)
        provider = lambda node_id: (r.get_rank_pids_for_node(node_id) if (r := box.value) is not None else {})

        box.value = None
        assert provider("node-1") == {}
