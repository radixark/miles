"""Tests for event_analyzer rules/checksum_compare shared primitives."""

from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import compare_flat_dicts, flatten_nested


class TestCompareFlatDicts:
    def test_identical_dicts_no_issue(self) -> None:
        """Identical flat dicts yield no mismatch."""
        assert list(compare_flat_dicts({"w": "a"}, {"w": "a"}, "x", "y")) == []

    def test_differing_value_reported(self) -> None:
        """A differing value is reported with both labels and values."""
        issues = list(compare_flat_dicts({"w": "a"}, {"w": "b"}, "x", "y"))
        assert len(issues) == 1
        assert issues[0].key == "w"
        assert issues[0].value_a == "a"
        assert issues[0].value_b == "b"

    def test_missing_key_marked(self) -> None:
        """A key present only on one side is reported as <missing> on the other."""
        issues = list(compare_flat_dicts({"w": "a"}, {}, "x", "y"))
        assert issues[0].value_b == "<missing>"


class TestFlattenNested:
    def test_nested_dict_flattened(self) -> None:
        """Nested dicts flatten to dot-separated keys."""
        assert flatten_nested({"state": {0: {"x": "h"}}}, prefix="opt") == {"opt.state.0.x": "h"}

    def test_list_indexed(self) -> None:
        """List entries flatten to bracket-indexed keys."""
        assert flatten_nested({"p": ["a", "b"]}, prefix="r") == {"r.p[0]": "a", "r.p[1]": "b"}
