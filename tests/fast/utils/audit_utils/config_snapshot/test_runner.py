import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from collections.abc import Callable
from pathlib import Path

import yaml

from miles.utils.audit_utils.config_snapshot.compact_codec import CompactConfigSnapshotCase
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotVariant
from miles.utils.audit_utils.config_snapshot.runner import ConfigSnapshotMismatch, ConfigSnapshotTestRunner
from miles.utils.test_utils.snapshot import SNAPSHOT_DIRECTORY, SNAPSHOT_RECORD_DIR_ENV_VAR, SNAPSHOT_UPDATE_ENV_VAR


_VARIANT = ConfigSnapshotVariant(hardware="h200", wandb_available=False)


def test_snapshot_update_writes_overrides_without_modifying_shared_templates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_record: Callable
) -> None:
    """Snapshot updates retain the manually maintained template and record changed values in the case."""
    bases = tmp_path / SNAPSHOT_DIRECTORY / "runtime_config/base.yaml"
    bases.parent.mkdir(parents=True)
    original = "templates:\n  default:\n    args:\n      value: old\n"
    bases.write_text(original)
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
    monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")
    runner = ConfigSnapshotTestRunner.create(test="tests/e2e/test_training.py", repo_root=tmp_path, variant=_VARIANT)
    runner.storage.write(make_record(config={"value": "new"}))

    runner.finish(returncode=0)

    case = CompactConfigSnapshotCase.model_validate(yaml.safe_load(runner.golden.read_text()))
    process = next(iter(case.processes.values()))
    assert process.template == "default"
    assert process.overrides.model_dump() == {"set": {"/args/value": "new"}, "remove": []}
    assert bases.read_text() == original
    monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR)
    bases.write_text("templates:\n  default:\n    args:\n      value: new\n")
    with pytest.raises(ConfigSnapshotMismatch, match="does not match"):
        runner.finish(returncode=0)


class TestSnapshotAttemptLifecycle:
    def test_capture_requires_an_explicit_record_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capture cannot silently write raw records into an implicit working directory."""
        monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)
        with pytest.raises(ValueError, match="MILES_SNAPSHOT_RECORD_DIR"):
            ConfigSnapshotTestRunner.create(test="tests/e2e/test_example.py", repo_root=tmp_path, variant=_VARIANT)

    def test_repeated_attempts_get_distinct_directories_and_start_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retries of one test cannot overwrite each other's raw evidence."""
        import json

        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
        first = ConfigSnapshotTestRunner.create(test="tests/e2e/test_example.py", repo_root=tmp_path, variant=_VARIANT)
        second = ConfigSnapshotTestRunner.create(
            test="tests/e2e/test_example.py", repo_root=tmp_path, variant=_VARIANT
        )

        assert first.directory != second.directory
        for runner in (first, second):
            assert runner.record_directory.is_dir()
            assert json.loads((runner.directory / "attempt.json").read_text())["completed"] is False

    def test_failed_execution_preserves_records_without_comparing_or_updating_a_golden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed child retains its raw evidence without interpreting partial snapshots."""
        import json

        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
        monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")
        runner = ConfigSnapshotTestRunner.create(
            test="tests/e2e/test_example.py", repo_root=tmp_path, variant=_VARIANT
        )
        invalid = runner.record_directory / "partial.json"
        invalid.write_text("incomplete json")

        runner.finish(returncode=7)

        assert invalid.read_text() == "incomplete json"
        assert not runner.golden.exists()
        assert json.loads((runner.directory / "attempt.json").read_text())["completed"] is False

    def test_success_without_records_completes_without_creating_an_empty_golden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tests that never launch training remain valid without synthetic empty snapshots."""
        import json

        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
        runner = ConfigSnapshotTestRunner.create(
            test="tests/e2e/test_example.py", repo_root=tmp_path, variant=_VARIANT
        )
        runner.finish(returncode=0)

        assert not runner.golden.exists()
        assert json.loads((runner.directory / "attempt.json").read_text())["completed"] is True

    @pytest.mark.parametrize("test", ["../outside.py", "tests/../outside.py"])
    def test_parent_traversal_is_rejected(self, tmp_path: Path, test: str) -> None:
        """A test identifier cannot escape the repository's snapshot tree."""
        with pytest.raises(ValueError, match="within the repository"):
            ConfigSnapshotTestRunner.golden_path(test=test, repo_root=tmp_path, variant=_VARIANT)

    def test_symlink_escape_is_rejected(self, tmp_path: Path) -> None:
        """A relative test path cannot redirect its golden through a symlink outside the snapshot tree."""
        root = tmp_path / "repo"
        directory = root / SNAPSHOT_DIRECTORY / "runtime_config/variants/h200/no-wandb"
        directory.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (directory / "escaped").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match="within the snapshot directory"):
            ConfigSnapshotTestRunner.golden_path(test="escaped/test.py", repo_root=root, variant=_VARIANT)


class TestMissingSnapshotEvidence:
    @pytest.mark.parametrize("update", [False, True])
    def test_existing_golden_requires_records_even_in_update_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, update: bool
    ) -> None:
        """A successful child cannot bypass an existing golden by emitting no configuration records."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
        if update:
            monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")
        else:
            monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR, raising=False)
        runner = ConfigSnapshotTestRunner.create(
            test="tests/e2e/test_example.py", repo_root=tmp_path, variant=_VARIANT
        )
        runner.golden.parent.mkdir(parents=True)
        runner.golden.write_text("preserve existing golden")

        with pytest.raises(ConfigSnapshotMismatch, match="No configuration snapshots were recorded"):
            runner.finish(returncode=0)

        assert runner.golden.read_text() == "preserve existing golden"


class TestSnapshotVariants:
    def test_hardware_and_tracking_variants_retain_independent_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_record: Callable
    ) -> None:
        """Normal hardware and credential differences cannot overwrite another variant's baseline."""
        import json

        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
        monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")
        runners = []
        for hardware in ("h200", "b200"):
            for wandb_available in (False, True):
                variant = ConfigSnapshotVariant(hardware=hardware, wandb_available=wandb_available)
                runner = ConfigSnapshotTestRunner.create(
                    test="tests/e2e/test_example.py", repo_root=tmp_path, variant=variant
                )
                runner.storage.write(make_record(config={"hardware": hardware, "use_wandb": wandb_available}))
                runner.finish(returncode=0)
                assert json.loads((runner.directory / "attempt.json").read_text())["variant"] == variant.model_dump()
                runners.append(runner)

        assert len({runner.golden for runner in runners}) == 4
        monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR)
        for runner in runners:
            runner.finish(returncode=0)

    def test_missing_variant_does_not_fall_back_to_a_canonical_golden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_record: Callable
    ) -> None:
        """A baseline without trusted variant provenance cannot satisfy a new variant's gate."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
        monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR, raising=False)
        canonical = tmp_path / SNAPSHOT_DIRECTORY / "runtime_config/tests/e2e/test_example.yaml"
        canonical.parent.mkdir(parents=True)
        canonical.write_text("unattributed baseline")
        runner = ConfigSnapshotTestRunner.create(
            test="tests/e2e/test_example.py", repo_root=tmp_path, variant=_VARIANT
        )
        runner.storage.write(make_record())

        with pytest.raises(ConfigSnapshotMismatch):
            runner.finish(returncode=0)

        assert canonical.read_text() == "unattributed baseline"
        assert not runner.golden.exists()

    @pytest.mark.parametrize(
        "fields", [{"hardware": "cuda", "wandb_available": False}, {"hardware": "h200", "wandb_available": "false"}]
    )
    def test_ambiguous_variant_values_are_rejected(self, fields: dict) -> None:
        """Unknown hardware and coerced credential flags cannot silently select a baseline."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ConfigSnapshotVariant.model_validate(fields)


class TestMissingVariantCapture:
    @pytest.mark.parametrize("existing", ["canonical", "other-hardware", "other-tracking"])
    @pytest.mark.parametrize("update", [False, True])
    def test_known_test_cannot_lose_capture_when_its_variant_has_no_golden(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str, update: bool
    ) -> None:
        """Known snapshot coverage survives variant migration without using another baseline for comparison."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
        if update:
            monkeypatch.setenv(SNAPSHOT_UPDATE_ENV_VAR, "1")
        else:
            monkeypatch.delenv(SNAPSHOT_UPDATE_ENV_VAR, raising=False)
        test = "tests/e2e/test_example.py"
        if existing == "canonical":
            baseline = tmp_path / SNAPSHOT_DIRECTORY / "runtime_config" / Path(test).with_suffix(".yaml")
        else:
            variant = ConfigSnapshotVariant(hardware="b200", wandb_available=False)
            if existing == "other-tracking":
                variant = ConfigSnapshotVariant(hardware="h200", wandb_available=True)
            baseline = ConfigSnapshotTestRunner.golden_path(test=test, repo_root=tmp_path, variant=variant)
        baseline.parent.mkdir(parents=True)
        baseline.write_text("not parsed as YAML: [")
        runner = ConfigSnapshotTestRunner.create(test=test, repo_root=tmp_path, variant=_VARIANT)

        with pytest.raises(ConfigSnapshotMismatch, match="No configuration snapshots were recorded"):
            runner.finish(returncode=0)

        assert not runner.golden.exists()
        assert baseline.read_text() == "not parsed as YAML: ["

    def test_unrelated_test_with_the_same_basename_does_not_require_capture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Coverage belongs to the complete test path rather than its filename alone."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path / "records"))
        baseline = ConfigSnapshotTestRunner.golden_path(
            test="tests/e2e/test_example.py", repo_root=tmp_path, variant=_VARIANT
        )
        baseline.parent.mkdir(parents=True)
        baseline.write_text("existing unrelated baseline")
        runner = ConfigSnapshotTestRunner.create(
            test="tests/fast-gpu/test_example.py", repo_root=tmp_path, variant=_VARIANT
        )

        runner.finish(returncode=0)

        assert not runner.golden.exists()
