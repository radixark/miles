import json
from argparse import Namespace

import pytest
import yaml
from pydantic import ValidationError
from tests.fast.charts.utils import NAMESPACE, REPO_ROOT, RUN_RELEASE_NAME, objects_of_kind, render_run, requires_helm
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import engine, router, trainer

from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    SECTION_OF_CATEGORY,
    InfraInfo,
    LaunchPlan,
    MooncakeInfo,
)

RUN_CHART_DIR = REPO_ROOT / "charts" / "miles-run"

MOONCAKE_INIT_KWARGS = {"master_server_address": "127.0.0.1:50051", "local_hostname": "localhost"}


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


def _mooncake_args(*, object_store_backend: str = "mooncake", **overrides: object) -> Namespace:
    return Namespace(
        object_store_backend=object_store_backend,
        mooncake_store_init_kwargs={**MOONCAKE_INIT_KWARGS, **overrides},
    )


def _mooncake_argv(**overrides: object) -> list[str]:
    kwargs = {**MOONCAKE_INIT_KWARGS, **overrides}
    return [
        "python",
        "train.py",
        "--object-store-backend",
        "mooncake",
        "--mooncake-store-init-kwargs",
        json.dumps(kwargs),
        "--lr",
        "1e-6",
    ]


def _rewritten_kwargs(argv: list[str]) -> dict[str, object]:
    return json.loads(argv[argv.index("--mooncake-store-init-kwargs") + 1])


class TestMooncakePlanOfArgs:
    def test_reads_the_backend_and_the_port_the_run_configured(self):
        """The chart publishes this port, so reading it wrong points every client at a closed socket."""
        plan = MooncakeInfo.plan_of_args(_mooncake_args())

        assert plan is not None
        assert plan.port == 50051

    def test_ignores_a_run_that_names_another_object_store(self):
        """A run on the default backend must not gain a master StatefulSet it never talks to."""
        assert MooncakeInfo.plan_of_args(_mooncake_args(object_store_backend="ray")) is None

    def test_refuses_an_address_that_carries_no_port(self):
        """A bare host would render a Service on port zero, which nothing can dial."""
        with pytest.raises(AssertionError, match="carries no port"):
            MooncakeInfo.plan_of_args(_mooncake_args(master_server_address="127.0.0.1"))

    def test_refuses_a_mooncake_run_that_configured_no_kwargs(self):
        """There is no address to rewrite then, and a pod's loopback master would hang every client."""
        args = Namespace(object_store_backend="mooncake", mooncake_store_init_kwargs=None)

        with pytest.raises(AssertionError, match="is missing"):
            MooncakeInfo.plan_of_args(args)


class TestMooncakeWithClusterMaster:
    def test_points_the_master_address_at_the_in_cluster_service(self):
        """The launcher's own loopback address means nothing inside a pod, which would hang every client."""
        rewritten = MooncakeInfo.with_cluster_master(
            _mooncake_argv(),
            plan=MooncakeInfo.plan_of_args(_mooncake_args()),
            host="mooncake.myns.svc.cluster.local",
        )

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "mooncake.myns.svc.cluster.local:50051"

    def test_a_run_that_never_named_the_flag_is_given_it(self):
        """The backend moves a run onto this store without being asked, so the flag is missing exactly
        when the run did not choose mooncake. Every pod parses this argv on its own, so omitting it
        leaves each of them looking for a master on its own loopback."""
        plan = MooncakeInfo.plan_of_args(_mooncake_args())
        argv = ["python", "train.py", "--num-rollout", "1"]

        rewritten = MooncakeInfo.with_cluster_master(argv, plan=plan, host="mooncake.myns.svc.cluster.local")

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "mooncake.myns.svc.cluster.local:50051"
        assert rewritten[: len(argv)] == argv

    def test_keeps_the_port_the_run_configured(self):
        """The Service publishes the port the values carry, so rewriting the host must not move the port."""
        plan = MooncakeInfo.plan_of_args(_mooncake_args(master_server_address="1.2.3.4:60000"))

        rewritten = MooncakeInfo.with_cluster_master(
            _mooncake_argv(master_server_address="1.2.3.4:60000"), plan=plan, host="host"
        )

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "host:60000"

    def test_keeps_every_other_init_kwarg(self):
        """The kwargs are rewritten as a whole, and a dropped one changes how the store is built."""
        rewritten = MooncakeInfo.with_cluster_master(
            _mooncake_argv(), plan=MooncakeInfo.plan_of_args(_mooncake_args()), host="host"
        )

        assert _rewritten_kwargs(rewritten)["local_hostname"] == "localhost"

    def test_leaves_the_rest_of_the_argv_untouched(self):
        """Only the address is cluster-specific; every other argument is the experiment itself."""
        rewritten = MooncakeInfo.with_cluster_master(
            _mooncake_argv(), plan=MooncakeInfo.plan_of_args(_mooncake_args()), host="host"
        )

        assert rewritten[:5] == _mooncake_argv()[:5]
        assert rewritten[-2:] == ["--lr", "1e-6"]

    def test_passes_a_non_mooncake_run_through_unchanged(self):
        """A run that never asked for mooncake has no address to rewrite, and no kwargs to invent."""
        argv = ["python", "train.py", "--lr", "1e-6"]

        assert MooncakeInfo.with_cluster_master(argv, plan=None, host="host") == argv


@requires_helm
class TestMooncakeServiceNameCoupling:
    def test_the_host_it_builds_is_the_service_the_chart_renders(self):
        """The pods dial this name; a chart rename would leave the launcher pointing at nothing."""
        objects = render_run("--set", "run.mooncake.enabled=true")
        services = [
            obj for obj in objects_of_kind(objects, "Service") if obj["metadata"]["name"].endswith("mooncake-master")
        ]

        assert len(services) == 1
        assert MooncakeInfo.master_object_name(RUN_RELEASE_NAME) == services[0]["metadata"]["name"]
        assert MooncakeInfo.master_service_host(RUN_RELEASE_NAME, NAMESPACE) == (
            f"{services[0]['metadata']['name']}.{NAMESPACE}.svc.cluster.local"
        )
