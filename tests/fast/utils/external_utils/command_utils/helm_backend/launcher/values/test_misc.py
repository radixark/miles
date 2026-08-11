import pytest
import yaml
from pydantic import ValidationError
from tests.fast.charts.utils import REPO_ROOT
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import engine, router, trainer

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    SECTION_OF_CATEGORY,
    InfraInfo,
    LaunchPlan,
)

RUN_CHART_DIR = REPO_ROOT / "charts" / "miles-run"


class TestSectionOf:
    def test_sends_a_pool_that_declares_nothing_to_the_static_workers(self):
        """A router is never healed per cell, so a pool_id would only add indirection."""
        assert SECTION_OF_CATEGORY[router().category] == "staticWorkers"

    def test_keeps_a_single_cell_engine_a_pool(self):
        """The provider recognises cells by LeaderWorkerSet labels, which a plain workload would not carry."""
        assert SECTION_OF_CATEGORY[engine(num_cells=1, gpus_per_engine=8).category] == "inferenceEngines"

    def test_sends_a_pool_that_declares_itself_an_engine_to_the_engines(self):
        """An engine group is restarted as a unit and so needs a pool_id."""
        assert SECTION_OF_CATEGORY[engine().category] == "inferenceEngines"

    def test_sends_a_pool_that_declares_itself_a_trainer_to_the_trainer_engines(self):
        """Trainers are served over rpc rather than launched as a command, and heal per dp group."""
        assert SECTION_OF_CATEGORY[trainer().category] == "trainerEngines"


class TestLaunchPlan:
    def test_rejects_a_field_the_chart_would_never_read(self):
        """A misspelled plan field would otherwise be dropped, and the run would launch mis-shaped."""
        with pytest.raises(ValidationError):
            LaunchPlan(
                run_id="260101-000000-000",
                state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
                release="r",
                namespace="rl",
                orchestrator_command=[],
                worker_argv=[],
                node_local_rooot="/scratch",
            )


def _resolved(tmp_path, *files: dict) -> str:
    paths = []
    for index, values in enumerate(files):
        path = tmp_path / f"infra-{index}.yaml"
        path.write_text(yaml.safe_dump(values))
        paths.append(str(path))
    return InfraInfo.shared_root(InfraInfo.load(RUN_CHART_DIR, paths))


class TestSharedRootOf:
    def test_falls_back_to_the_chart_defaults_when_no_file_says_otherwise(self, tmp_path):
        """The chart's own values.yaml is the single source of these defaults; Python must not carry a copy."""
        assert _resolved(tmp_path, {}) == "/cluster-storage/miles_data"

    def test_hangs_the_runs_off_the_configured_sub_path(self, tmp_path):
        """Runs live beside the other miles data on the cluster filesystem, not at its root."""
        values = {"infra": {"sharedStorage": {"mountPath": "/mnt/x"}, "paths": {"runsSubPath": "teamdata"}}}

        assert _resolved(tmp_path, values) == "/mnt/x/teamdata"

    def test_an_empty_sub_path_puts_the_runs_at_the_mount_root(self, tmp_path):
        """A cluster that dedicates the whole volume to miles must not be forced into a subdirectory."""
        values = {"infra": {"sharedStorage": {"mountPath": "/mnt/x"}, "paths": {"runsSubPath": ""}}}

        assert _resolved(tmp_path, values) == "/mnt/x"

    def test_a_nulled_section_drops_the_chart_default_as_helm_does(self, tmp_path):
        """helm deletes a key a values file nulls, so a launcher that re-defaulted it would pick another path."""
        values = {"infra": {"sharedStorage": {"mountPath": "/mnt/x"}, "paths": None}}

        assert _resolved(tmp_path, values) == "/mnt/x"

    def test_the_last_file_that_names_a_value_wins(self, tmp_path):
        """helm applies --values files in order, and the launcher must resolve the same run directory it renders."""
        first = {"infra": {"sharedStorage": {"mountPath": "/mnt/a"}}}
        second = {"infra": {"paths": {"runsSubPath": "b"}}}

        assert _resolved(tmp_path, first, second) == "/mnt/a/b"

    def test_a_file_that_sets_one_key_keeps_the_rest_of_the_section(self, tmp_path):
        """A shallow merge would drop the chart's storage type and leave the run with no volume at all."""
        values = {"infra": {"sharedStorage": {"mountPath": "/mnt/x"}}}
        loaded = InfraInfo.load(RUN_CHART_DIR, [_written(tmp_path, values)])

        assert (loaded.shared_storage.type, loaded.shared_storage.mount_path) == ("hostPath", "/mnt/x")


class TestLoadInfraValues:
    def test_rejects_a_section_the_charts_do_not_define(self, tmp_path):
        """helm would reject the same file at install time, and failing here says so before anything runs."""
        values = {"infra": {"sharedStorag": {"mountPath": "/mnt/x"}}}

        with pytest.raises(ValueError, match="sharedStorag"):
            InfraInfo.load(RUN_CHART_DIR, [_written(tmp_path, values)])

    def test_reads_the_defaults_the_chart_ships(self, tmp_path):
        """Every launch merges onto these, so a chart whose defaults stopped validating breaks every run."""
        loaded = InfraInfo.load(RUN_CHART_DIR, [])

        assert loaded.image.repository == "radixark/miles"
        assert loaded.shared_storage.mount_path == "/cluster-storage"


def _written(tmp_path, values: dict) -> str:
    path = tmp_path / "infra.yaml"
    path.write_text(yaml.safe_dump(values))
    return str(path)
