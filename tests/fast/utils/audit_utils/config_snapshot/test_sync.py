import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from collections.abc import Callable
from pathlib import Path


from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotTestAttempt, ConfigSnapshotVariant
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.config_snapshot.sync import _collect_snapshots
from miles.utils.test_utils.snapshot import SNAPSHOT_DIRECTORY


_VARIANT = ConfigSnapshotVariant(hardware="h200", wandb_available=False)


def test_completed_unit_test_without_records_does_not_block_training_snapshot(
    tmp_path: Path, make_record: Callable
) -> None:
    """An uploaded archive can omit an empty records directory without losing another test's snapshot."""
    raw = tmp_path / "raw"
    empty = raw / "unit" / "attempt"
    populated = raw / "training" / "attempt"
    _write_attempt(directory=empty, test="tests/fast-gpu/test_unit.py")
    _write_attempt(directory=populated, test="tests/e2e/test_training.py")
    ConfigSnapshotStorage(directory=populated / "records").write(make_record(config={"value": "retained"}))
    _write_bases(repo_root=tmp_path)

    snapshots = _collect_snapshots(directory=raw, repo_root=tmp_path)

    assert list(snapshots) == [
        tmp_path / SNAPSHOT_DIRECTORY / "runtime_config/variants/h200/no-wandb/tests/e2e/test_training.yaml"
    ]
    assert "value: retained" in next(iter(snapshots.values()))
    assert (empty / "attempt.json").is_file()
    assert not (empty / "records").exists()


def test_empty_and_populated_completed_attempts_for_one_test_still_disagree(
    tmp_path: Path, make_record: Callable
) -> None:
    """An empty completed attempt cannot be filtered out to hide a same-test coverage difference."""
    raw = tmp_path / "raw"
    empty = raw / "test" / "empty"
    populated = raw / "test" / "populated"
    for directory in (empty, populated):
        _write_attempt(directory=directory, test="tests/e2e/test_training.py")
    ConfigSnapshotStorage(directory=populated / "records").write(make_record(config={"value": "retained"}))
    _write_bases(repo_root=tmp_path)

    with pytest.raises(ValueError, match="Completed attempts disagree"):
        _collect_snapshots(directory=raw, repo_root=tmp_path)


def _write_attempt(*, directory: Path, test: str) -> None:
    directory.mkdir(parents=True)
    (directory / "attempt.json").write_text(
        ConfigSnapshotTestAttempt(variant=_VARIANT, test=test, completed=True).model_dump_json()
    )


def _write_bases(*, repo_root: Path) -> None:
    path = repo_root / SNAPSHOT_DIRECTORY / "runtime_config/base.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("templates:\n  default: {}\n")


class TestCompletedAttemptSelection:
    def test_incomplete_records_without_a_successful_attempt_block_sync(
        self, tmp_path: Path, make_record: Callable
    ) -> None:
        """Partial captures cannot be promoted to golden snapshots after a failed run."""
        attempt = tmp_path / "raw" / "attempt"
        _write_attempt(directory=attempt, test="tests/e2e/test_training.py")
        (attempt / "attempt.json").write_text(
            ConfigSnapshotTestAttempt(
                variant=_VARIANT, test="tests/e2e/test_training.py", completed=False
            ).model_dump_json()
        )
        ConfigSnapshotStorage(directory=attempt / "records").write(make_record())

        with pytest.raises(ValueError, match="No completed attempt"):
            _collect_snapshots(directory=tmp_path / "raw", repo_root=tmp_path)

    def test_failed_attempts_do_not_override_successful_retry_contents(
        self, tmp_path: Path, make_record: Callable
    ) -> None:
        """A successful retry replaces failed evidence without requiring its partial values to agree."""
        root = tmp_path / "raw"
        failed, succeeded = root / "failed", root / "succeeded"
        for directory in (failed, succeeded):
            _write_attempt(directory=directory, test="tests/e2e/test_training.py")
        (failed / "attempt.json").write_text(
            ConfigSnapshotTestAttempt(
                variant=_VARIANT, test="tests/e2e/test_training.py", completed=False
            ).model_dump_json()
        )
        ConfigSnapshotStorage(directory=failed / "records").write(make_record(config={"value": "partial"}))
        ConfigSnapshotStorage(directory=succeeded / "records").write(make_record(config={"value": "complete"}))

        snapshots = _collect_snapshots(directory=root, repo_root=tmp_path)

        assert len(snapshots) == 1
        assert "complete" in next(iter(snapshots.values()))
        assert "partial" not in next(iter(snapshots.values()))

    def test_an_archive_without_completed_records_is_rejected(self, tmp_path: Path) -> None:
        """An empty artifact download cannot report a successful golden refresh."""
        with pytest.raises(ValueError, match="No completed snapshot records"):
            _collect_snapshots(directory=tmp_path, repo_root=tmp_path)


class TestVariantSelection:
    def test_same_test_with_distinct_variants_syncs_distinct_contents(
        self, tmp_path: Path, make_record: Callable
    ) -> None:
        """Successful platform and tracking variants retain their actual differing configurations."""
        for hardware in ("h200", "b200"):
            for available in (False, True):
                variant = ConfigSnapshotVariant(hardware=hardware, wandb_available=available)
                directory = tmp_path / "raw" / hardware / str(available)
                directory.mkdir(parents=True)
                (directory / "attempt.json").write_text(
                    ConfigSnapshotTestAttempt(
                        test="tests/e2e/test_training.py", variant=variant, completed=True
                    ).model_dump_json()
                )
                ConfigSnapshotStorage(directory=directory / "records").write(
                    make_record(config={"hardware": hardware, "use_wandb": available})
                )

        snapshots = _collect_snapshots(directory=tmp_path / "raw", repo_root=tmp_path)

        assert len(snapshots) == 4
        assert len(set(snapshots.values())) == 4

    def test_success_on_other_hardware_cannot_complete_a_failed_variant(
        self, tmp_path: Path, make_record: Callable
    ) -> None:
        """An incomplete hardware variant cannot borrow success from another platform."""
        for hardware, completed in (("h200", True), ("b200", False)):
            directory = tmp_path / "raw" / hardware
            directory.mkdir(parents=True)
            (directory / "attempt.json").write_text(
                ConfigSnapshotTestAttempt(
                    test="tests/e2e/test_training.py",
                    variant=ConfigSnapshotVariant(hardware=hardware, wandb_available=False),
                    completed=completed,
                ).model_dump_json()
            )
            ConfigSnapshotStorage(directory=directory / "records").write(make_record())

        with pytest.raises(ValueError, match="No completed attempt.*b200"):
            _collect_snapshots(directory=tmp_path / "raw", repo_root=tmp_path)

    def test_legacy_attempt_without_variant_is_rejected(self, tmp_path: Path) -> None:
        """Old raw records cannot acquire fabricated platform or credential provenance."""
        from pydantic import ValidationError

        (tmp_path / "attempt.json").write_text('{"test":"tests/e2e/test_training.py","completed":true}')
        with pytest.raises(ValidationError, match="variant"):
            _collect_snapshots(directory=tmp_path, repo_root=tmp_path)


class TestKnownCoverage:
    @pytest.mark.parametrize("existing", ["canonical", "same-variant", "other-variant"])
    def test_empty_known_test_blocks_a_mixed_archive_before_any_writes(
        self, tmp_path: Path, make_record: Callable, existing: str
    ) -> None:
        """Another successful test cannot hide missing records for an already covered test."""
        from miles.utils.audit_utils.config_snapshot.runner import ConfigSnapshotTestRunner
        from miles.utils.audit_utils.config_snapshot.sync import sync

        test = "tests/e2e/test_training.py"
        if existing == "canonical":
            baseline = tmp_path / SNAPSHOT_DIRECTORY / "runtime_config" / Path(test).with_suffix(".yaml")
        else:
            variant = (
                _VARIANT
                if existing == "same-variant"
                else ConfigSnapshotVariant(hardware="b200", wandb_available=False)
            )
            baseline = ConfigSnapshotTestRunner.golden_path(test=test, repo_root=tmp_path, variant=variant)
        baseline.parent.mkdir(parents=True)
        baseline.write_text("not parsed as YAML: [")
        empty = tmp_path / "raw" / "empty"
        populated = tmp_path / "raw" / "populated"
        _write_attempt(directory=empty, test=test)
        _write_attempt(directory=populated, test="tests/e2e/test_other.py")
        ConfigSnapshotStorage(directory=populated / "records").write(make_record())

        with pytest.raises(AssertionError, match="No configuration snapshots were recorded"):
            sync(directory=tmp_path / "raw", repo_root=tmp_path)

        assert baseline.read_text() == "not parsed as YAML: ["
        other = ConfigSnapshotTestRunner.golden_path(
            test="tests/e2e/test_other.py", repo_root=tmp_path, variant=_VARIANT
        )
        assert not other.exists()
