import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from tests.utils.soak.core import utils as utils_module
from tests.utils.soak.core.utils import (
    API_SERVER_PORT,
    assert_fresh_dump_dir,
    compute_base_url,
    compute_release_of_config,
    create_soak_config,
    evidence_directory,
    get_dumps_root,
    recording_error,
    resolve_dump_dir,
)

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME

_RUN_ID = "260926-120000-000"


class TestResolveDumpDir:
    def test_the_dump_dir_hangs_off_the_configured_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cluster says where dumps go through the environment its infra file sets."""
        monkeypatch.setenv("MILES_TEST_DUMPS_ROOT", str(tmp_path / "dumps"))

        assert resolve_dump_dir("scenario_x", run_id="run-a") == str(tmp_path / "dumps" / "run-a" / "scenario_x")

    def test_an_empty_configured_root_falls_back_to_the_shared_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unset variable and one set to nothing both mean the cluster configured no root."""
        monkeypatch.setenv("MILES_TEST_DUMPS_ROOT", "")
        made: list[Path] = []
        monkeypatch.setattr(utils_module.os, "makedirs", lambda path, exist_ok: made.append(Path(path)))

        assert resolve_dump_dir("scenario_x", run_id="run-a") == "/node_public/dumps/run-a/scenario_x"
        assert made == [Path("/node_public/dumps/run-a/scenario_x")]

    def test_two_runs_of_one_test_do_not_share_a_dump_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run id in the path is what stops one run's cleanup deleting another's dumps."""
        monkeypatch.setenv("MILES_TEST_DUMPS_ROOT", str(tmp_path))

        assert resolve_dump_dir("scenario_x", run_id="run-a") != resolve_dump_dir("scenario_x", run_id="run-b")

    def test_the_dump_directory_exists_when_it_is_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callers write into the returned path without creating it themselves."""
        monkeypatch.setenv("MILES_TEST_DUMPS_ROOT", str(tmp_path / "dumps"))

        assert Path(resolve_dump_dir("scenario_x", run_id="run-a")).is_dir()


class TestGetDumpsRoot:
    def test_a_relative_root_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A relative root would land dumps in whatever directory the runner happened to start in."""
        monkeypatch.setenv("MILES_TEST_DUMPS_ROOT", "dumps")

        with pytest.raises(ValueError, match="absolute"):
            get_dumps_root()

    def test_an_unset_root_is_the_shared_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without configuration dumps go to the node-shared directory."""
        monkeypatch.delenv("MILES_TEST_DUMPS_ROOT", raising=False)

        assert get_dumps_root() == Path("/node_public/dumps")


class TestAssertFreshDumpDir:
    def test_a_missing_directory_is_created(self, tmp_path: Path) -> None:
        """A new run gets its dump directory created."""
        dump_dir = tmp_path / "run" / "scenario"

        assert_fresh_dump_dir(dump_dir)

        assert dump_dir.is_dir()

    def test_an_existing_empty_directory_is_accepted(self, tmp_path: Path) -> None:
        """An empty directory holds no stale artifacts to confuse the checkers."""
        assert_fresh_dump_dir(tmp_path)

    def test_a_directory_with_artifacts_is_refused(self, tmp_path: Path) -> None:
        """Stale events from an earlier run must not be read as this run's evidence."""
        (tmp_path / "events").mkdir()

        with pytest.raises(ValueError, match="existing artifacts"):
            assert_fresh_dump_dir(tmp_path)


class TestEvidenceDirectory:
    def test_the_evidence_lives_beside_the_dump_dir_under_a_unique_name(self, tmp_path: Path) -> None:
        """Evidence is kept outside the dump dir and never shared between two soaks."""
        dump_dir = tmp_path / "scenario"

        first, second = evidence_directory(dump_dir), evidence_directory(dump_dir)

        assert first.parent == second.parent == tmp_path / "scenario-soak"
        assert first != second
        assert not first.is_relative_to(dump_dir)


class TestComputeBaseUrl:
    def test_a_ray_run_answers_on_localhost_at_the_soak_api_port(self) -> None:
        """The soak observes a Ray run through the api server port it launched with."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, run_id=_RUN_ID)

        assert compute_base_url(config) == f"http://localhost:{API_SERVER_PORT}"


class TestComputeReleaseOfConfig:
    def test_the_release_is_named_by_run_component_and_instance(self) -> None:
        """Cleanup targets the release this config launched and no other."""
        config = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES,
            namespace="rl",
            run_id=_RUN_ID,
            deploy_component=DeployComponent.INFERENCE,
            deploy_instance_id="b",
        )

        assert compute_release_of_config(config) == f"{CHART_NAME}-{_RUN_ID}-inference-b"
        assert compute_release_of_config(replace(config, deploy_instance_id="c")) != compute_release_of_config(config)


class TestCreateSoakConfig:
    def test_every_ray_soak_owns_a_fresh_submission_id(self) -> None:
        """Each Ray soak gets its own submission id so teardown only stops its own job."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY, run_id=_RUN_ID, ray_submission_id="theirs")

        first, second = create_soak_config(config), create_soak_config(config)

        assert first.ray_submission_id.startswith("miles-soak-")
        assert first.ray_submission_id != second.ray_submission_id
        assert replace(first, ray_submission_id=None) == replace(config, ray_submission_id=None)

    def test_a_kubernetes_config_is_returned_unchanged(self) -> None:
        """A Kubernetes soak is owned through its release, not a Ray submission."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace="rl", run_id=_RUN_ID)

        assert create_soak_config(config) is config


class TestRecordingError:
    def test_an_exception_is_recorded_under_its_key_and_swallowed(self) -> None:
        """One failed read becomes an observation error without aborting the others."""
        errors: dict[str, str] = {"cells": "earlier"}

        with recording_error(errors, "pods"):
            raise RuntimeError("kubectl down")

        assert errors == {"cells": "earlier", "pods": repr(RuntimeError("kubectl down"))}

    def test_a_cancellation_is_not_swallowed(self) -> None:
        """Cancelling an observation must propagate instead of turning into an error entry."""
        errors: dict[str, str] = {}

        with pytest.raises(asyncio.CancelledError):
            with recording_error(errors, "pods"):
                raise asyncio.CancelledError

        assert errors == {}

    def test_a_clean_block_records_nothing(self) -> None:
        """A successful read leaves the error map untouched."""
        errors: dict[str, str] = {}

        with recording_error(errors, "pods"):
            pass

        assert errors == {}
