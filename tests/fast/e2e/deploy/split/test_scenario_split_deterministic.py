from dataclasses import dataclass, field
from typing import Any

import pytest
from examples.infra_features.split_deployment.address_book import (
    DEFAULT_TRAINER_ID,
    INIT_EXPECTED_NUM_CELLS_FLAG,
    RunAddressBook,
)
from tests.e2e.deploy.conftest_deploy.common import utils as deploy_utils
from tests.e2e.deploy.conftest_deploy.split import scenario_split_deterministic as scenario
from tests.e2e.deploy.conftest_deploy.split import split_deployment
from tests.e2e.deploy.conftest_deploy.split.split_deployment import RunDeployment
from tests.e2e.ft.conftest_ft import app as ft_app
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, RunSideRequest
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.fast.train_args import (
    FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON,
    FLAGS_A_SPLIT_RUN_MAY_DIFFER_FROM_AN_UNSPLIT_ONE_ON,
    shared_argv,
    value_of,
    values_after,
)

from miles.ray.specs.inference import INFERENCE_CONTROLLER_ADDR_FLAG
from miles.ray.specs.train import TRAINER_CONTROLLER_ADDRS_FLAG
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, OBJECT_STORE_BACKEND_FLAG
from miles.utils.workers.types import ClusterBackend, DeployComponent

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"
DUMP_DIR: str = "/dumps/one-side"


@pytest.fixture
def mode() -> FTTestMode:
    return scenario._MODE


@pytest.fixture
def address_book() -> RunAddressBook:
    return RunAddressBook(run_id=RUN_ID, run_uuid=RUN_UUID, namespace=NAMESPACE)


@pytest.fixture
def deployments(mode: FTTestMode) -> list[RunDeployment]:
    return scenario._build_deployments(_request(mode))


class TestMode:
    def test_the_scenario_deploys_more_than_one_group_of_engines(self, mode):
        """A single engine deployment is a shape the unsplit baseline already covers."""
        assert mode.rollout_num_engines > 1

    def test_the_scenario_shares_no_gpus_between_the_trainer_and_the_engines(self, mode):
        """Separate deployments cannot be colocated, and a mode that asked for it would fail deep inside helm."""
        assert not mode.colocate


class TestBuildDeployments:
    def test_the_engines_of_a_run_are_deployed_one_group_at_a_time(self, deployments, mode):
        """Several engine deployments registering into one run is the whole subject of this scenario."""
        assert len(_deployments_of(deployments, DeployComponent.INFERENCE)) == mode.rollout_num_engines

    def test_every_engine_deployment_is_named_apart_from_the_others(self, deployments):
        """Two engine deployments of one name install one release, and the run loses half its engines."""
        instance_ids = [one.deploy_instance_id for one in _deployments_of(deployments, DeployComponent.INFERENCE)]

        assert None not in instance_ids
        assert len(set(instance_ids)) == len(instance_ids)

    def test_the_engine_deployments_together_carry_what_the_baseline_runs_alone(self, deployments, mode):
        """A target with fewer engines than the baseline would be compared against a different run."""
        declared = [
            int(value_of(one.train_args, scenario.ROLLOUT_NUM_GPUS_FLAG))
            for one in _deployments_of(deployments, DeployComponent.INFERENCE)
        ]

        assert sum(declared) == mode.total_rollout_gpus

    def test_an_engine_deployment_is_given_the_gpus_of_exactly_one_engine(self, deployments, mode):
        """A deployment holding two engines is the shape the baseline already runs, and splits nothing."""
        for one in _deployments_of(deployments, DeployComponent.INFERENCE):
            assert value_of(one.train_args, scenario.ROLLOUT_NUM_GPUS_FLAG) == str(mode.rollout_gpus_per_engine)
            assert value_of(one.train_args, scenario.ROLLOUT_NUM_GPUS_PER_ENGINE_FLAG) == str(
                mode.rollout_gpus_per_engine
            )

    def test_the_driving_deployment_still_counts_every_engine_the_run_registers(self, deployments, mode):
        """It waits for as many engine cells as its own arguments declare, and would start on half a fleet."""
        driver = _deployments_of(deployments, DeployComponent.PRIMARY)[0]

        assert value_of(driver.train_args, scenario.ROLLOUT_NUM_GPUS_FLAG) == str(mode.total_rollout_gpus)

    def test_only_the_driving_deployment_is_told_how_many_engines_to_wait_for(self, deployments, mode):
        """A deployment that installs no engines of its own has to be told, and would start on a bare run."""
        told = {one.deploy_component for one in deployments if INIT_EXPECTED_NUM_CELLS_FLAG in one.train_args}
        driver = _deployments_of(deployments, DeployComponent.PRIMARY)[0]

        assert told == {DeployComponent.PRIMARY}
        assert value_of(driver.train_args, INIT_EXPECTED_NUM_CELLS_FLAG) == str(mode.rollout_num_engines)

    def test_only_the_engine_deployments_are_told_where_to_register(self, deployments):
        """Every other deployment holds the controller itself and refuses to be pointed at one."""
        told = {one.deploy_component for one in deployments if INFERENCE_CONTROLLER_ADDR_FLAG in one.train_args}

        assert told == {DeployComponent.INFERENCE}

    def test_every_engine_deployment_registers_into_the_controller_of_this_run(self, deployments, address_book):
        """An address that only looks right is an engine deployment that joins some other run, or none."""
        expected = value_of(address_book.inference_controller_addr_arg(), INFERENCE_CONTROLLER_ADDR_FLAG)

        for one in _deployments_of(deployments, DeployComponent.INFERENCE):
            assert value_of(one.train_args, INFERENCE_CONTROLLER_ADDR_FLAG) == expected

    def test_only_the_driving_deployment_is_told_where_the_trainer_is(self, deployments):
        """The deployment that carries the trainer reaches it in its own process and refuses the flag."""
        told = {one.deploy_component for one in deployments if TRAINER_CONTROLLER_ADDRS_FLAG in one.train_args}

        assert told == {DeployComponent.PRIMARY}

    def test_the_driving_deployment_dials_the_trainer_release_of_this_run(self, deployments, address_book):
        """The driver reaches the trainer by name alone, so a name that drifted reaches nothing at all."""
        driver = _deployments_of(deployments, DeployComponent.PRIMARY)[0]
        expected = address_book.trainer_controller_addrs_arg(
            deploy_instance_id_of_trainer_id={DEFAULT_TRAINER_ID: None}
        )

        assert values_after(driver.train_args, TRAINER_CONTROLLER_ADDRS_FLAG) == values_after(
            expected, TRAINER_CONTROLLER_ADDRS_FLAG
        )

    def test_the_run_is_driven_by_the_deployment_installed_last(self, deployments):
        """Installing it earlier would block on a run whose workers are not there yet."""
        assert deployments[-1].deploy_component is DeployComponent.PRIMARY

    def test_the_trainer_is_deployed_before_the_script_that_drives_it(self, deployments):
        """The driving deployment is handed the trainer's address, so the trainer has to be on its way."""
        components = [one.deploy_component for one in deployments]

        assert components.index(DeployComponent.TRAINER) < components.index(DeployComponent.PRIMARY)

    def test_every_deployment_redeems_its_references_at_one_object_store(self, deployments):
        """Deployments that disagree on the master hand each other references nothing can read back."""
        addresses = {value_of(one.train_args, MOONCAKE_INIT_KWARGS_FLAG) for one in deployments}

        assert len(addresses) == 1
        assert all(value_of(one.train_args, OBJECT_STORE_BACKEND_FLAG) == "mooncake" for one in deployments)

    def test_the_deployments_agree_on_everything_the_run_itself_declares(self, deployments):
        """Only what a deployment carries may differ; a drifted model or batch shape trains something else."""
        shared = [
            shared_argv(one.train_args, differing_flags=FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON)
            for one in deployments
        ]

        assert all(one == shared[0] for one in shared)


class TestBuildArgs:
    def test_the_baseline_installs_a_run_that_runs_its_own_object_store(self, mode):
        """The baseline is one release, so nothing outside it names the master it should dial."""
        baseline = scenario._build_baseline_args(mode, DUMP_DIR)

        assert value_of(baseline, OBJECT_STORE_BACKEND_FLAG) == "mooncake"

    def test_the_two_sides_differ_in_nothing_but_how_they_are_deployed(self, mode, deployments):
        """A bitwise comparison across a second difference would prove nothing about deployment."""
        driver = _deployments_of(deployments, DeployComponent.PRIMARY)[0]

        baseline = scenario._build_baseline_args(mode, DUMP_DIR, True, _request(mode).config)

        assert _shared_argv(driver.train_args) == _shared_argv(baseline)

    def test_the_baseline_is_grouped_under_the_run_it_is_launched_as(self, mode, monkeypatch):
        """A group named after a config nobody launched files the baseline's metrics under a run that never ran."""
        monkeypatch.setenv("WANDB_API_KEY", "unused-in-this-test")
        monkeypatch.delenv("GITHUB_COMMIT_NAME", raising=False)

        baseline = scenario._build_baseline_args(mode, DUMP_DIR, True, _request(mode).config)

        assert value_of(baseline, "--wandb-group") == RUN_ID

    def test_the_run_trains_without_weight_decay(self, mode):
        """Weight decay moves weights on its own, which would let a run that learned nothing pass the moved gate."""
        assert value_of(scenario._build_args(mode, DUMP_DIR), "--weight-decay") == "0"

    def test_a_colocated_mode_is_refused(self, mode):
        """Colocation shares gpus between the very deployments this scenario installs apart."""
        with pytest.raises(AssertionError, match="colocates them on shared gpus"):
            scenario._build_args(_colocated(mode), DUMP_DIR)

    def test_a_mode_without_engines_is_refused(self, mode):
        """There would be no engine deployment left to install, and the scenario would test nothing."""
        with pytest.raises(AssertionError, match="engines to deploy"):
            scenario._build_args(_without_engines(mode), DUMP_DIR)


def _deployments_of(deployments: list[RunDeployment], component: DeployComponent) -> list[RunDeployment]:
    return [one for one in deployments if one.deploy_component is component]


def _request(mode: FTTestMode, *, side: str = TARGET_SIDE) -> RunSideRequest:
    return RunSideRequest(
        side=side,
        mode=mode,
        train_args=scenario._build_args(mode, DUMP_DIR),
        dump_dir=DUMP_DIR,
        config=ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID, run_uuid=RUN_UUID
        ),
        enable_dumper=True,
    )


def _colocated(mode: FTTestMode) -> FTTestMode:
    return FTTestMode(
        model_name=mode.model_name,
        model_hf_repo=mode.model_hf_repo,
        megatron_model_type=mode.megatron_model_type,
        num_cells=mode.num_cells,
        train_gpus_per_node=mode.train_gpus_per_node,
        rollout_num_engines=mode.rollout_num_engines,
        rollout_gpus_per_engine=mode.rollout_gpus_per_engine,
        colocate=True,
        ft_components=("rollout",),
        parallel_args=mode.parallel_args,
    )


def _without_engines(mode: FTTestMode) -> FTTestMode:
    return FTTestMode(
        model_name=mode.model_name,
        model_hf_repo=mode.model_hf_repo,
        megatron_model_type=mode.megatron_model_type,
        num_cells=mode.num_cells,
        train_gpus_per_node=mode.train_gpus_per_node,
        parallel_args=mode.parallel_args,
    )


def _shared_argv(train_args: str) -> list[str]:
    return shared_argv(train_args, differing_flags=FLAGS_A_SPLIT_RUN_MAY_DIFFER_FROM_AN_UNSPLIT_ONE_ON)


# =========================== how the pipeline dispatches ==========================


@dataclass
class _Pipeline:
    launched: list[DeployComponent] = field(default_factory=list)
    instance_ids: list[str | None] = field(default_factory=list)
    split_sides: list[str] = field(default_factory=list)
    unsplit_sides: list[tuple[str, str]] = field(default_factory=list)
    compared_after: int | None = None

    def run_ci(self) -> None:
        scenario._create_app_and_run_ci()[1]()

    def build_deployments(self, request: RunSideRequest) -> list[RunDeployment]:
        self.split_sides.append(request.side)
        return [
            RunDeployment(deploy_component=DeployComponent.TRAINER, train_args=request.train_args),
            RunDeployment(
                deploy_component=DeployComponent.INFERENCE, train_args=request.train_args, deploy_instance_id="e0"
            ),
            RunDeployment(
                deploy_component=DeployComponent.INFERENCE, train_args=request.train_args, deploy_instance_id="e1"
            ),
            RunDeployment(deploy_component=DeployComponent.PRIMARY, train_args=request.train_args),
        ]

    def compare(self, dump_dir: str, mode: FTTestMode) -> None:
        self.compared_after = len(self.launched)

    def record(self, config: ExecuteTrainConfig) -> None:
        self.launched.append(config.deploy_component)
        self.instance_ids.append(config.deploy_instance_id)


@pytest.fixture
def pipeline(monkeypatch, tmp_path) -> _Pipeline:
    recorded = _Pipeline()
    run_pipeline = ft_app.run_pipeline

    def run_unsplit(request: RunSideRequest) -> None:
        recorded.unsplit_sides.append((request.side, request.train_args))
        ft_app.run_one_release(request)

    def run_pipeline_without_release(**kwargs: Any) -> None:
        run_pipeline(**kwargs, release_side=lambda _request: None)

    monkeypatch.setattr(deploy_utils, "assert_cluster_can_deploy_runs", lambda config: None)
    monkeypatch.setattr(scenario, "_build_args", _fake_target_args)
    monkeypatch.setattr(scenario, "_build_baseline_args", _fake_baseline_args)
    monkeypatch.setattr(scenario, "_build_deployments", recorded.build_deployments)
    monkeypatch.setattr(scenario, "_compare", recorded.compare)
    monkeypatch.setattr(ft_app, "resolve_dump_dir", lambda test_name: str(tmp_path / test_name))
    monkeypatch.setattr(ft_app, "prepare", lambda mode: None)
    monkeypatch.setattr(ft_app, "run_pipeline", run_pipeline_without_release)
    monkeypatch.setattr(command_utils, "default_config", _pipeline_config)
    monkeypatch.setattr(
        ft_app, "run_training", lambda *, train_args, mode, dump_dir=None, config: recorded.record(config)
    )
    monkeypatch.setattr(split_deployment, "run_training", lambda *, train_args, mode, config: recorded.record(config))
    monkeypatch.setattr(split_deployment, "run_one_release", run_unsplit)
    monkeypatch.setattr(split_deployment, "Helm", _FakeHelm())

    return recorded


def _fake_target_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return "--some-flag some-value "


def _fake_baseline_args(
    mode: FTTestMode, dump_dir: str, enable_dumper: bool = True, config: ExecuteTrainConfig | None = None
) -> str:
    return f"--some-flag some-value --its-own-object-store --run-id-of {None if config is None else config.run_id} "


def _pipeline_config() -> ExecuteTrainConfig:
    return ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID)


class _FakeHelm:
    @staticmethod
    def get_manifest(release: str, namespace: str) -> object | None:
        return object()

    @staticmethod
    def uninstall(*, release: str, namespace: str) -> None:
        return None


class TestTheScenarioPipeline:
    def test_only_the_target_side_is_installed_as_several_deployments(self, pipeline):
        """A baseline that was also split would compare two split runs and prove nothing about splitting."""
        pipeline.run_ci()

        assert pipeline.split_sides == [TARGET_SIDE]
        assert [side for side, _ in pipeline.unsplit_sides] == [BASELINE_SIDE]

    def test_the_baseline_is_installed_with_the_arguments_its_own_builder_composed(self, pipeline):
        """A side installed on whatever the harness happened to carry would train something else entirely."""
        pipeline.run_ci()

        assert [train_args for _, train_args in pipeline.unsplit_sides] == [
            _fake_baseline_args(scenario._MODE, DUMP_DIR, True, _pipeline_config())
        ]

    def test_the_baseline_is_built_against_the_run_it_is_installed_as(self, pipeline):
        """Built against a config of its own, it would file this run's metrics under a run nobody launched."""
        pipeline.run_ci()

        [(_, train_args)] = pipeline.unsplit_sides

        assert f"--run-id-of {RUN_ID} " in train_args

    def test_the_run_reaches_the_cluster_one_deployment_at_a_time_in_this_order(self, pipeline):
        """The baseline is one release; the target's parts install in the order the run needs them installed."""
        pipeline.run_ci()

        assert pipeline.launched == [
            DeployComponent.ALL,
            DeployComponent.TRAINER,
            DeployComponent.INFERENCE,
            DeployComponent.INFERENCE,
            DeployComponent.PRIMARY,
        ]

    def test_every_engine_deployment_of_the_target_is_named_apart_from_the_others(self, pipeline):
        """Two engine deployments installed under one name leave the run with half the engines it counted on."""
        pipeline.run_ci()

        assert pipeline.instance_ids == [None, None, "e0", "e1", None]

    def test_the_two_sides_are_compared_once_both_have_run(self, pipeline):
        """A comparison run before the second side would read one side's dumps as both."""
        pipeline.run_ci()

        assert pipeline.compared_after == len(pipeline.launched)

    def test_a_cluster_that_cannot_carry_the_releases_fails_before_deploying_anything(self, pipeline, monkeypatch):
        """An environment without kubernetes fails loudly rather than installing half a run into helm."""
        monkeypatch.setattr(deploy_utils, "assert_cluster_can_deploy_runs", _refuse_cluster)

        with pytest.raises(AssertionError, match="no cluster here"):
            pipeline.run_ci()

        assert not pipeline.launched


def _refuse_cluster(config: ExecuteTrainConfig) -> None:
    raise AssertionError("no cluster here")
