import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from typing import Any

import yaml

from miles.utils.test_utils.snapshot_yaml import dump_snapshot_delta, dump_snapshot_yaml, expand_snapshot_delta


class TestSnapshotDelta:
    @pytest.mark.parametrize(
        ("base", "actual"),
        [
            ("", ""),
            ("", "new"),
            ("old", ""),
            ("same\n", "same\n"),
            ("a\nb\nc\n", "before\na\nchanged\nc\nafter\n"),
            ("a\nb\n", "a\nb"),
            ("a\nb", "a\nb\n"),
            ("中文\n🙂\n", "中文\n另一行\n🙂\n"),
        ],
    )
    def test_deltas_preserve_every_byte_including_final_newlines(self, base: str, actual: str) -> None:
        """Text deltas reconstruct insertions, deletions and newline boundaries without loss."""
        delta = dump_snapshot_delta(base=base, actual=actual, base_name="base.yaml")

        assert expand_snapshot_delta(base=base, delta=delta) == actual

    def test_a_changed_base_is_rejected_before_applying_edits(self) -> None:
        """A stale baseline cannot silently reinterpret stored edit offsets."""
        delta = dump_snapshot_delta(base="a\n", actual="b\n", base_name="base.yaml")

        with pytest.raises(AssertionError, match="base has changed"):
            expand_snapshot_delta(base="changed\n", delta=delta)

    @pytest.mark.parametrize(
        "edits",
        [
            [{"at": -1, "delete": 0, "insert": ""}],
            [{"at": 0, "delete": 3, "insert": ""}],
            [{"at": True, "delete": 0, "insert": ""}],
            [{"at": 0, "delete": 0, "insert": []}],
            [{"at": 0, "delete": 1, "insert": ""}, {"at": 0, "delete": 0, "insert": ""}],
            [{"at": 0, "delete": 0, "insert": "", "unknown": 1}],
        ],
    )
    def test_malformed_or_overlapping_edits_are_rejected(self, edits: list[dict[str, Any]]) -> None:
        """Invalid edit coordinates and payloads cannot corrupt a reconstructed baseline."""
        payload = yaml.safe_load(dump_snapshot_delta(base="a\nb\n", actual="", base_name="base.yaml"))
        payload["edits"] = edits

        with pytest.raises(AssertionError):
            expand_snapshot_delta(base="a\nb\n", delta=yaml.safe_dump(payload))


class TestSnapshotYaml:
    def test_shared_mapping_compaction_preserves_types_and_unique_fields(self) -> None:
        """YAML sharing must preserve per-record overrides and scalar type distinctions."""
        common = {"long_value": "x" * 100, "another_value": [1, 2, None]}
        values = {"a": common | {"unique": True}, "b": common | {"unique": 1}, "c": common | {"unique": "1"}}

        restored = yaml.safe_load(dump_snapshot_yaml(values))

        assert restored == values
        assert type(restored["a"]["unique"]) is bool
        assert type(restored["b"]["unique"]) is int
        assert type(restored["c"]["unique"]) is str

    def test_multiline_and_yaml_looking_strings_remain_strings(self) -> None:
        """YAML serialization must not coerce textual nulls, booleans or dates."""
        values = {"items": ["null", "true", "2026-01-01", "007", "line\nnext\n", "中文🙂"]}

        assert yaml.safe_load(dump_snapshot_yaml(values)) == values
