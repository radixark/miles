import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from pathlib import Path

import yaml

from miles.utils.test_utils.snapshot import SNAPSHOT_UPDATE_ENV_VAR, assert_scenario_snapshots, dump_snapshot
from miles.utils.test_utils.snapshot_yaml import expand_snapshot_delta


class TestScenarioSnapshotStorage:
    def test_updates_store_a_compact_variant_and_remove_only_its_obsolete_formats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An updated scenario has one current representation that reconstructs from its declared baseline."""
        base = {f"item-{index:03d}": {"value": f"stable-value-{index:03d}"} for index in range(200)}
        variant = base | {"changed": {"value": "new-value"}}
        stale_yaml = tmp_path / "variant.yaml"
        stale_diff = tmp_path / "variant.diff"
        unrelated = tmp_path / "unrelated.yaml"
        for path in (stale_yaml, stale_diff, unrelated):
            path.write_text("old\n")
        monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")

        assert_scenario_snapshots(
            snapshots={"base": dump_snapshot(base), "variant": dump_snapshot(variant)},
            bases={"variant": "base"},
            directory=tmp_path,
        )

        delta = (tmp_path / "variant.delta.yaml").read_text()
        baseline = (tmp_path / "base.yaml").read_text()
        assert len(delta.encode()) < len(baseline.encode())
        assert yaml.safe_load(baseline) == base
        assert yaml.safe_load(delta)["base"] == "base"
        assert yaml.safe_load(expand_snapshot_delta(base=baseline, delta=delta)) == variant
        assert not stale_yaml.exists()
        assert not stale_diff.exists()
        assert unrelated.read_text() == "old\n"

    def test_read_only_comparison_reports_every_failure_without_changing_any_baseline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One failing scenario cannot hide later mismatches or alter stored snapshots."""
        monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR, raising=False)
        paths = [tmp_path / name for name in ("first.yaml", "second.yaml", "second.diff")]
        for path in paths:
            path.write_text("old\n")

        with pytest.raises(AssertionError) as error:
            assert_scenario_snapshots(
                snapshots={"first": dump_snapshot({"value": 1}), "second": dump_snapshot({"value": 2})},
                bases={},
                directory=tmp_path,
            )

        assert "snapshot scenario first" in str(error.value)
        assert "snapshot scenario second" in str(error.value)
        assert all(path.read_text() == "old\n" for path in paths)
