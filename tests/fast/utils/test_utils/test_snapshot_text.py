import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pathlib import Path


from miles.utils.test_utils.snapshot import SNAPSHOT_UPDATE_ENV_VAR, assert_matches_snapshot
from miles.utils.test_utils.snapshot_text import compact_snapshot_text, read_snapshot
from miles.utils.test_utils.snapshot_yaml import dump_snapshot_delta


class TestSnapshotFiles:
    def test_compacted_text_expands_relative_to_its_own_directory(self, tmp_path: Path) -> None:
        """A compact snapshot locates its base beside the file rather than in the working directory."""
        base = "stable line\n" * 200
        actual = base + "changed\n"
        (tmp_path / "base.yaml").write_text(base)
        snapshot = tmp_path / "variant.yaml"
        snapshot.write_text(compact_snapshot_text(base=base, actual=actual, base_name="base.yaml"))

        assert snapshot.stat().st_size < len(actual.encode())
        assert read_snapshot(snapshot) == actual

    def test_small_snapshots_stay_literal_when_delta_metadata_would_expand_them(self) -> None:
        """Compaction never increases storage for a tiny snapshot."""
        assert compact_snapshot_text(base="a", actual="b", base_name="base.yaml") == "b"

    def test_absolute_base_paths_are_rejected(self, tmp_path: Path) -> None:
        """Transported snapshots cannot depend on machine-specific absolute baseline paths."""
        snapshot = tmp_path / "variant.yaml"
        delta = dump_snapshot_delta(base="base", actual="changed", base_name=str(tmp_path / "base.yaml"))
        snapshot.write_text("# miles-snapshot-delta-v1\n" + delta)

        with pytest.raises(AssertionError, match="relative path"):
            read_snapshot(snapshot)

    def test_nested_text_deltas_are_rejected(self, tmp_path: Path) -> None:
        """A delta chain cannot silently treat another delta document as baseline text."""
        base = "# miles-snapshot-delta-v1\ninvalid"
        (tmp_path / "base.yaml").write_text(base)
        snapshot = tmp_path / "variant.yaml"
        snapshot.write_text(
            "# miles-snapshot-delta-v1\n" + dump_snapshot_delta(base=base, actual="x", base_name="base.yaml")
        )

        with pytest.raises(AssertionError, match="Nested text snapshot"):
            read_snapshot(snapshot)

    def test_missing_baseline_reports_actual_content_without_writing_it(self, tmp_path: Path) -> None:
        """A failed assertion provides reviewable actual data and leaves baseline creation explicit."""
        path = tmp_path / "missing.yaml"
        with pytest.raises(AssertionError) as error:
            assert_matches_snapshot(snapshot=path, actual="actual\n", subject="subject", update=False)

        assert "BEGIN ACTUAL missing.yaml" in str(error.value)
        assert "actual\n" in str(error.value)
        assert not path.exists()

    def test_explicit_read_only_mode_overrides_the_update_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callers disabling updates cannot overwrite goldens through an inherited environment flag."""
        path = tmp_path / "value.yaml"
        path.write_text("old\n")
        monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")

        with pytest.raises(AssertionError):
            assert_matches_snapshot(snapshot=path, actual="new\n", subject="subject", update=False)
        assert path.read_text() == "old\n"

    def test_explicit_update_creates_parent_directories(self, tmp_path: Path) -> None:
        """Approved baseline creation works for a new nested scenario directory."""
        path = tmp_path / "nested" / "value.yaml"
        assert_matches_snapshot(snapshot=path, actual="new\n", subject="subject", update=True)

        assert path.read_text() == "new\n"
