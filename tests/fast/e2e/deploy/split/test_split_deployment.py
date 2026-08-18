from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.split import split_deployment
from tests.e2e.deploy.conftest_deploy.split.split_deployment import (
    RunDeployment,
    create_launch_of_mode,
    run_split_training,
    run_split_training_into,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import ClusterBackend, DeployComponent

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"
SENTINEL_NAME: str = "written-by-a-deployment.txt"

MODE: FTTestMode = FTTestMode(
    model_name="Qwen3-0.6B",
    model_hf_repo="Qwen/Qwen3-0.6B",
    megatron_model_type="qwen3-0.6B",
    num_cells=2,
    train_gpus_per_node=4,
    rollout_num_engines=2,
    rollout_gpus_per_engine=1,
    parallel_args="--context-parallel-size 2",
)


@pytest.fixture
def config() -> ExecuteTrainConfig:
    return ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID, run_uuid=RUN_UUID
    )


def _release(deploy_component: DeployComponent, deploy_instance_id: str | None = None) -> str:
    return ReleaseName(
        run_id=RUN_ID, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id
    ).serialize()


class TestRunSplitTraining:
    def test_it_installs_the_workers_before_the_script_that_drives_them(self, config, deployment_launches, fake_helm):
        """The driving launch blocks until the run finishes, so anything installed after it never arrives."""
        _install(config=config, launches=deployment_launches)

        assert [one.deploy_component for one in deployment_launches.records] == [
            DeployComponent.TRAINER,
            DeployComponent.INFERENCE,
            DeployComponent.INFERENCE,
            DeployComponent.PRIMARY,
        ]

    def test_every_deployment_of_one_run_carries_the_same_run_uuid(self, config, deployment_launches, fake_helm):
        """Nothing but the run uuid joins releases that separate launches installed."""
        _install(config=config, launches=deployment_launches)

        assert {one.run_uuid for one in deployment_launches.records} == {RUN_UUID}

    def test_a_run_whose_launches_would_each_invent_an_identity_is_refused(self, deployment_launches, fake_helm):
        """Deployments agreeing on nothing but their release names are as many runs as there are launches."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID)

        with pytest.raises(AssertionError, match="joined by nothing but the run uuid"):
            _install(config=config, launches=deployment_launches)

    def test_a_run_whose_launches_name_no_namespace_is_refused(self, deployment_launches, fake_helm):
        """Releases installed into whatever namespace is default reach each other under names nothing holds."""
        config = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES, namespace="", run_id=RUN_ID, run_uuid=RUN_UUID
        )

        with pytest.raises(AssertionError, match="a namespace is half of every such name"):
            _install(config=config, launches=deployment_launches)

    def test_every_deployment_keeps_the_arguments_it_was_built_with(self, config, deployment_launches, fake_helm):
        """A deployment's arguments describe only what it carries, so they must not be swapped between launches."""
        _install(config=config, launches=deployment_launches)

        assert [(one.train_args, one.deploy_instance_id) for one in deployment_launches.records] == [
            ("trainer", None),
            ("inference-a", "a"),
            ("inference-b", "b"),
            ("primary", None),
        ]

    def test_it_uninstalls_the_deployments_that_outlive_the_run(self, config, deployment_launches, fake_helm):
        """A workers-only release has no training to finish, so it stays up until the launcher takes it down."""
        _install(config=config, launches=deployment_launches)

        assert set(fake_helm.uninstalled) == {
            _release(DeployComponent.TRAINER),
            _release(DeployComponent.INFERENCE, "a"),
            _release(DeployComponent.INFERENCE, "b"),
        }

    def test_it_leaves_the_driving_release_to_the_launcher_that_installed_it(
        self, config, deployment_launches, fake_helm
    ):
        """The driving release follows the run to its end and tears itself down; a second uninstall races it."""
        _install(config=config, launches=deployment_launches)

        assert _release(DeployComponent.PRIMARY) not in fake_helm.uninstalled

    def test_it_uninstalls_them_even_when_the_run_fails(self, config, deployment_launches, fake_helm):
        """A failed run that leaked its engine releases would hold the cluster's gpus until someone noticed."""
        deployment_launches.fail_on = DeployComponent.PRIMARY

        with pytest.raises(RuntimeError):
            _install(config=config, launches=deployment_launches)

        assert len(fake_helm.uninstalled) == 3

    def test_it_uninstalls_a_deployment_whose_own_launch_failed(self, config, deployment_launches, fake_helm):
        """helm may have installed the release before the launch raised, so it is torn down too."""
        deployment_launches.fail_on = DeployComponent.INFERENCE

        with pytest.raises(RuntimeError):
            _install(config=config, launches=deployment_launches)

        assert _release(DeployComponent.INFERENCE, "a") in fake_helm.uninstalled

    def test_a_release_that_refuses_to_be_uninstalled_does_not_strand_the_others(
        self, config, deployment_launches, fake_helm
    ):
        """One release nobody can take down would otherwise leave every earlier one holding gpus for good."""
        fake_helm.fail_to_uninstall = _release(DeployComponent.INFERENCE, "b")

        _install(config=config, launches=deployment_launches)

        assert set(fake_helm.uninstalled) == {
            _release(DeployComponent.TRAINER),
            _release(DeployComponent.INFERENCE, "a"),
        }

    def test_a_failed_uninstall_does_not_replace_the_failure_that_ended_the_run(
        self, config, deployment_launches, fake_helm
    ):
        """The run's own failure is what has to be read; a teardown error on top of it hides the cause."""
        deployment_launches.fail_on = DeployComponent.PRIMARY
        fake_helm.fail_to_uninstall = _release(DeployComponent.INFERENCE, "b")

        with pytest.raises(RuntimeError, match="the primary deployment failed"):
            _install(config=config, launches=deployment_launches)

    def test_it_refuses_deployments_that_drive_the_run_before_the_workers(
        self, config, deployment_launches, fake_helm
    ):
        """Catching the order here is cheaper than watching a run wait for workers nobody installed."""
        with pytest.raises(AssertionError, match="blocks until the run finishes"):
            _install(config=config, launches=deployment_launches, deployments=list(reversed(_deployments())))

        assert not deployment_launches.records

    def test_it_refuses_deployments_that_carry_no_orchestration_script(self, config, deployment_launches, fake_helm):
        """Workers alone train nothing, and the pipeline would report a pass for a run that never started."""
        workers_only = [one for one in _deployments() if one.deploy_component is not DeployComponent.PRIMARY]

        with pytest.raises(AssertionError, match="exactly one deployment"):
            _install(config=config, launches=deployment_launches, deployments=workers_only)

    def test_it_refuses_a_deployment_that_carries_the_whole_run_by_itself(
        self, config, deployment_launches, fake_helm
    ):
        """An `all` deployment drives the run and would pass every check below while splitting nothing."""
        deployments = [RunDeployment(deploy_component=DeployComponent.ALL, train_args="everything")]

        with pytest.raises(AssertionError, match="carries a whole run in one release"):
            _install(config=config, launches=deployment_launches, deployments=deployments)

        assert not deployment_launches.records

    def test_it_refuses_two_deployments_that_would_install_one_release(self, config, deployment_launches, fake_helm):
        """Unnamed engine deployments overwrite each other, leaving a run with half the engines it counted on."""
        deployments = [
            RunDeployment(deploy_component=DeployComponent.INFERENCE, train_args="inference-a"),
            RunDeployment(deploy_component=DeployComponent.INFERENCE, train_args="inference-b"),
            RunDeployment(deploy_component=DeployComponent.PRIMARY, train_args="primary"),
        ]

        with pytest.raises(AssertionError, match="install one release over the other"):
            _install(config=config, launches=deployment_launches, deployments=deployments)

    def test_it_refuses_to_name_a_deployment_a_run_holds_only_one_of(self, config, deployment_launches, fake_helm):
        """An instance id reads as one of several, and a run drives exactly one orchestration script."""
        deployments = [
            RunDeployment(deploy_component=DeployComponent.TRAINER, train_args="trainer"),
            RunDeployment(deploy_component=DeployComponent.PRIMARY, train_args="primary", deploy_instance_id="a"),
        ]

        with pytest.raises(AssertionError, match="tells it apart from nothing"):
            _install(config=config, launches=deployment_launches, deployments=deployments)

    def test_it_fails_when_helm_does_not_know_a_release_it_just_installed(
        self, config, deployment_launches, fake_helm
    ):
        """A silently absent release turns into a run that waits forever, which is far harder to read."""
        fake_helm.known = False

        with pytest.raises(AssertionError, match="helm does not know it"):
            _install(config=config, launches=deployment_launches)


class TestRunSplitTrainingInto:
    def test_it_trains_every_deployment_under_the_mode_the_side_runs(self, config, tmp_path, launches, fake_helm):
        """A deployment launched under another mode would install a topology the comparison never asked for."""
        _run(config=config, dump_dir=str(tmp_path / "target"))

        assert [one.mode for one in launches.records] == [MODE] * 4

    def test_it_clears_what_an_earlier_run_left_before_the_first_deployment_starts(
        self, config, tmp_path, launches, fake_helm
    ):
        """A deployment writing into a stale dump directory would be compared against another run's numbers."""
        dump_dir = _seeded_dump_dir(tmp_path)
        launches.observe_dump_dir = dump_dir

        _run(config=config, dump_dir=dump_dir)

        assert launches.sentinel_seen == [False, False, False, False]

    def test_it_clears_the_dump_directory_once_and_leaves_it_to_the_run_after_that(
        self, config, tmp_path, launches, fake_helm
    ):
        """Clearing it again between deployments would delete what the deployments before it already wrote."""
        dump_dir = _seeded_dump_dir(tmp_path)
        launches.observe_dump_dir = dump_dir
        launches.write_sentinel = True

        _run(config=config, dump_dir=dump_dir)

        assert launches.sentinel_seen == [False, True, True, True]

    def test_a_run_whose_dump_directory_does_not_exist_yet_still_starts(self, config, tmp_path, launches, fake_helm):
        """The first run under a test name has nothing to clear, which is no reason to refuse to deploy."""
        _run(config=config, dump_dir=str(tmp_path / "never-written"))

        assert len(launches.records) == 4


def _install(
    *, config: ExecuteTrainConfig, launches: "_DeploymentRecorder", deployments: list[RunDeployment] | None = None
) -> None:
    run_split_training(
        deployments=_deployments() if deployments is None else deployments, launch=launches.launch, config=config
    )


def _run(*, config: ExecuteTrainConfig, dump_dir: str) -> None:
    run_split_training_into(
        deployments=_deployments(), launch=create_launch_of_mode(MODE), config=config, dump_dir=dump_dir
    )


def _seeded_dump_dir(tmp_path: Path) -> str:
    dump_dir = tmp_path / "target"
    dump_dir.mkdir()
    (dump_dir / SENTINEL_NAME).write_text("an earlier run wrote this")
    return str(dump_dir)


def _deployments() -> list[RunDeployment]:
    return [
        RunDeployment(deploy_component=DeployComponent.TRAINER, train_args="trainer"),
        RunDeployment(deploy_component=DeployComponent.INFERENCE, train_args="inference-a", deploy_instance_id="a"),
        RunDeployment(deploy_component=DeployComponent.INFERENCE, train_args="inference-b", deploy_instance_id="b"),
        RunDeployment(deploy_component=DeployComponent.PRIMARY, train_args="primary"),
    ]


@dataclass
class _DeploymentLaunch:
    train_args: str
    deploy_component: DeployComponent
    deploy_instance_id: str | None
    run_uuid: str | None


@dataclass
class _DeploymentRecorder:
    records: list[_DeploymentLaunch] = field(default_factory=list)
    fail_on: DeployComponent | None = None

    def launch(self, train_args: str, config: ExecuteTrainConfig) -> None:
        self.records.append(
            _DeploymentLaunch(
                train_args=train_args,
                deploy_component=config.deploy_component,
                deploy_instance_id=config.deploy_instance_id,
                run_uuid=config.run_uuid,
            )
        )
        if config.deploy_component is self.fail_on:
            raise RuntimeError(f"the {config.deploy_component.value} deployment failed")


@pytest.fixture
def deployment_launches() -> _DeploymentRecorder:
    return _DeploymentRecorder()


@dataclass
class _Launch:
    train_args: str
    mode: FTTestMode
    deploy_component: DeployComponent


@dataclass
class _LaunchRecorder:
    records: list[_Launch] = field(default_factory=list)
    observe_dump_dir: str | None = None
    write_sentinel: bool = False
    sentinel_seen: list[bool] = field(default_factory=list)

    def run_training(self, *, train_args: str, mode: FTTestMode, config: ExecuteTrainConfig) -> None:
        self.records.append(_Launch(train_args=train_args, mode=mode, deploy_component=config.deploy_component))
        if self.observe_dump_dir is None:
            return

        sentinel = Path(self.observe_dump_dir) / SENTINEL_NAME
        self.sentinel_seen.append(sentinel.exists())
        if self.write_sentinel:
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("this deployment wrote it")


@pytest.fixture
def launches(monkeypatch) -> _LaunchRecorder:
    recorder = _LaunchRecorder()
    monkeypatch.setattr(split_deployment, "run_training", recorder.run_training)
    return recorder


@dataclass
class _FakeHelm:
    known: bool = True
    fail_to_uninstall: str | None = None
    uninstalled: list[str] = field(default_factory=list)

    def get_manifest(self, release: str, namespace: str) -> object | None:
        return object() if self.known else None

    def uninstall(self, *, release: str, namespace: str) -> None:
        if release == self.fail_to_uninstall:
            raise RuntimeError(f"helm refused to uninstall {release}")
        self.uninstalled.append(release)


@pytest.fixture
def fake_helm(monkeypatch) -> _FakeHelm:
    helm = _FakeHelm()
    monkeypatch.setattr(split_deployment, "Helm", helm)
    return helm
