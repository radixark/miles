import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from argparse import Namespace
from pathlib import Path


from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


class TestCaptureState:
    @pytest.mark.parametrize("mode", ["not_ci", "disabled", "no_directory"])
    def test_reconfiguring_capture_off_does_not_leak_the_previous_process_state(
        self, capture_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """A later disabled process cannot append records under an earlier process's identity."""
        if mode == "not_ci":
            capture_args.ci_test = False
        elif mode == "disabled":
            capture_args.ci_disable_config_snapshot = True
        else:
            monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR)
        ConfigSnapshotDumper.configure(args=capture_args, source=SimpleProcessIdentity(component="rollout_executor"))

        ConfigSnapshotDumper.dump(stage="later_stage", config={"value": 9})

        records = ConfigSnapshotStorage(directory=tmp_path).read()
        assert [record.point.stage for record in records] == ["process_config"]

    def test_stage_counters_are_independent_and_invalid_stages_consume_no_indices(
        self, capture_args: Namespace, tmp_path: Path
    ) -> None:
        """Interleaved stages retain consecutive per-stage occurrence numbers."""
        ConfigSnapshotDumper.dump(stage="train_first_step", config={})
        with pytest.raises(ValueError, match="Invalid snapshot stage"):
            ConfigSnapshotDumper.dump(stage="../invalid", config={})
        ConfigSnapshotDumper.dump(stage="checkpoint_load", config={})
        ConfigSnapshotDumper.dump(stage="train_first_step", config={})

        records = ConfigSnapshotStorage(directory=tmp_path).read()
        assert sorted((record.point.stage, record.point.index) for record in records) == [
            ("checkpoint_load", 0),
            ("process_config", 0),
            ("train_first_step", 0),
            ("train_first_step", 1),
        ]

    def test_enabled_capture_requires_a_name(self, capture_args: Namespace) -> None:
        """Unnamed captures must fail before producing unidentifiable records."""
        capture_args.config_snapshot_name = None
        with pytest.raises(ValueError, match="Snapshot name is required"):
            ConfigSnapshotDumper.configure(args=capture_args, source=SimpleProcessIdentity(component="main"))
