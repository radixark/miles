import pydantic
import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values.utils import (
    LAYOUT,
    engine,
    router,
    session_server,
    trainer,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.values import pool_entry
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import PortEntry
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.worker_spec import BaseWorkerSpec, SchedulingSpec

PREPARE_CMD = "mkdir -p /scratch/dataset && rsync -a /cluster-storage/dataset/ /scratch/dataset"

PREPARE_LAYOUT = LaunchPlan(
    run_id="260101-000000-000",
    state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
    release="r",
    namespace="rl",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--foo", "bar"],
    prepare_cmd={"trainer": PREPARE_CMD},
)


def _prepared(section: str, spec: BaseWorkerSpec) -> list[str]:
    return build_values([spec], PREPARE_LAYOUT).as_values()["run"][section][0]["command"]


class TestBuildEntry:
    def test_refuses_a_spec_with_no_launch_mechanism(self):
        """A pool without a command or RPC server cannot launch a worker."""
        spec = BaseWorkerSpec(
            name="unsupported-worker",
            port_infos=[],
            env_var=lambda context: {},
            scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        )

        with pytest.raises(AssertionError, match="neither launched by a command nor served over rpc"):
            pool_entry.build_entry(spec, plan=LAYOUT, addresses={})

    def test_refuses_to_build_an_entry_for_a_spec_that_wants_no_cell(self):
        """Every entry renders at least one pod, so a zero-cell spec reaching conversion is a silent deploy."""
        with pytest.raises(AssertionError, match="cells"):
            pool_entry.build_entry(session_server(num_cells=0), plan=LAYOUT, addresses={})

    def test_replaces_only_the_node_rank_with_the_kubelet_placeholder(self):
        """Every pod of a group shares one command, so the rank must be the one part left to kubelet."""
        command = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--node-rank") + 1] == pool_entry._WORKER_INDEX_PLACEHOLDER
        assert command[command.index("--base-gpu-id") + 1] == "0"

    def test_points_a_master_port_at_the_group_leader(self):
        """A rank cannot know its leader's address until it is scheduled, but kubelet does."""
        command = build_values([engine()], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--dist-init-addr") + 1] == f"{pool_entry._LEADER_ADDRESS_PLACEHOLDER}:9000"

    def test_allows_a_command_that_never_mentions_its_rank(self):
        """Some engines do not take one; what matters is that no sentinel survives into the command."""
        spec = engine().model_copy(update={"launch_command": lambda ctx: "python -m sglang.launch_server"})

        command = build_values([spec], LAYOUT).as_values()["run"]["inferenceEngines"][0]["command"]

        assert str(pool_entry._WORKER_INDEX_SENTINEL) not in " ".join(command)


class TestPortEntry:
    @pytest.mark.parametrize("name", ["9000", "bad_name", "trailing-"])
    def test_refuses_a_name_that_is_not_an_iana_service_name(self, name: str):
        """The chart writes these straight into containerPort names, and the api server rejects the rest."""
        with pytest.raises(pydantic.ValidationError, match="String should match pattern"):
            PortEntry(name=name, port=9000)

    def test_accepts_the_names_the_launcher_computes(self):
        """A name compute_port_name produced must never be turned away by the model that carries it."""
        assert PortEntry(name="engine-info-boo", port=9000).name == "engine-info-boo"


class TestPrepareCmd:
    def test_a_trainer_runs_the_command_before_it_starts(self):
        """Every trainer rank reads the dataset every step, and the shared filesystem cannot serve that."""
        command = _prepared("trainerEngines", trainer())

        assert command[:2] == ["bash", "-c"]
        assert command[2].startswith(f"{PREPARE_CMD} && exec ")

    def test_an_engine_is_launched_without_the_command(self):
        """An sglang engine loads weights the trainer sends it, so the copy would move gigabytes nothing reads."""
        command = _prepared("inferenceEngines", engine())

        assert command[:2] != ["bash", "-c"]
        assert "/scratch/dataset" not in " ".join(command)

    def test_a_static_worker_is_launched_without_the_command(self):
        """A router holds no data of its own, and the copy would only delay the run's first request."""
        command = _prepared("staticWorkers", router())

        assert command[:2] != ["bash", "-c"]
        assert "/scratch/dataset" not in " ".join(command)

    def test_a_trainer_keeps_the_command_it_would_have_run(self):
        """The preparation prefixes the launch; a rewritten command would start the wrong worker."""
        command = _prepared("trainerEngines", trainer())

        assert command[2].endswith(" ".join(["--foo", "bar"]))

    def test_quotes_the_command_it_wraps(self):
        """An argument carrying json or spaces would otherwise be re-split by the shell that runs the pair."""
        spec = trainer(num_cells=1, gpus_per_cell=8)
        layout = PREPARE_LAYOUT.model_copy(update={"worker_argv": ["--kwargs", '{"a": 1}']})

        command = build_values([spec], layout).as_values()["run"]["trainerEngines"][0]["command"]

        assert command[2].endswith("""--kwargs \'{"a": 1}\'""")

    def test_refuses_a_trainer_whose_pods_can_share_a_node(self):
        """Two pods of one node would run the copy against the same node-local path at the same time."""
        with pytest.raises(AssertionError, match="can land on one node"):
            build_values([trainer(num_cells=1, gpus_per_cell=4)], PREPARE_LAYOUT).as_values()

    def test_accepts_a_trainer_that_takes_whole_nodes(self):
        """A pod holding every gpu of its node is the only pod there, so the copy cannot race itself."""
        command = _prepared("trainerEngines", trainer(num_cells=1, gpus_per_cell=8))

        assert command[:2] == ["bash", "-c"]

    def test_a_run_that_prepares_nothing_leaves_every_command_alone(self):
        """Most runs read from the shared filesystem directly, and a bash wrapper would hide their exit codes."""
        command = build_values([trainer()], LAYOUT).as_values()["run"]["trainerEngines"][0]["command"]

        assert command[:2] != ["bash", "-c"]
