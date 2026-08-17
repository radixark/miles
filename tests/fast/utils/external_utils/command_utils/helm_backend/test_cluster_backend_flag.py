from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.ray.test_wiring import stub_kubernetes_capability

from miles.ray import wiring
from miles.utils.external_utils.command_utils.base_backend import (
    _DEPLOY_COMPONENT_FLAG,
    CLUSTER_BACKEND_FLAG,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
)
from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan, MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.run_uuid import RUN_UUID_LENGTH
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec


def declared_cluster_backends(argv: list[str]) -> list[str]:
    return ArgvManipulator.get(argv, CLUSTER_BACKEND_FLAG)


def declared_deploy_components(argv: list[str]) -> list[str]:
    return ArgvManipulator.get(argv, _DEPLOY_COMPONENT_FLAG)


NAMESPACE = "rl"
RUN_ID = "260101-000000-000"
RUN_ID_SHORT_ENOUGH_FOR_FULL_OBJECT_NAMES = "260101-0000"


def _release(deploy_component: DeployComponent = DeployComponent.ALL) -> str:
    return ReleaseName(run_id=RUN_ID, deploy_component=deploy_component, deploy_instance_id=None).serialize()


SPLIT_RUN_UUID = "0123456789abcdef"


def _config(run_id: str = RUN_ID, deploy_component: DeployComponent = DeployComponent.ALL) -> ExecuteTrainConfig:
    return ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace=NAMESPACE,
        run_id=run_id,
        run_uuid=SPLIT_RUN_UUID if deploy_component.is_split() else None,
        deploy_component=deploy_component,
    )


def _request(train_args: str) -> ExecuteTrainRequest:
    return ExecuteTrainRequest(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd={},
        extra_manifests=[],
    )


def _router() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=30000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda context: "python -m router",
    )


@pytest.fixture(autouse=True)
def run_files_under_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # the shared root comes from the chart and points at the cluster's storage mount, which the
    # cpu lane neither has nor may create
    monkeypatch.setattr(
        entrypoint.InfraInfo,
        "shared_root",
        staticmethod(lambda _infra, *, namespace: str(tmp_path)),
    )


def launch_argv(
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_args: str,
    run_id: str = RUN_ID,
    deploy_component: DeployComponent = DeployComponent.ALL,
    recorded_releases: list[str] | None = None,
) -> list[str]:
    recorded: list[list[str]] = []

    def fake_compute_specs(args: Any) -> list[CommandWorkerSpec]:
        recorded.append(list(args.argv))
        return [_router()]

    def fake_parse_args() -> SimpleNamespace:
        argv = list(sys.argv[1:])
        declared = declared_deploy_components(argv)
        return SimpleNamespace(
            colocate=False,
            deploy_component=declared[-1] if declared else DeployComponent.ALL.value,
            deploy_instance_id=None,
            argv=argv,
            use_wandb=False,
            wandb_run_id=None,
        )

    def fake_upgrade(**kwargs: Any) -> None:
        if recorded_releases is not None:
            recorded_releases.append(kwargs["release"])

    def fake_render_upgrade(*, release: str, namespace: str, chart: Any, values_files: list[Any]) -> Manifest:
        return Manifest.parse("", namespace=namespace)

    monkeypatch.setattr(entrypoint, "compute_specs", fake_compute_specs)
    monkeypatch.setattr(entrypoint, "parse_args", fake_parse_args)
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_write_helm_values", lambda path, values: None)
    monkeypatch.setattr(entrypoint, "_compute_trainer_controller_addrs", lambda args, *, release, namespace: {})
    monkeypatch.setattr(Helm, "get_manifest", staticmethod(lambda release, namespace: None))
    monkeypatch.setattr(Helm, "render_upgrade", staticmethod(fake_render_upgrade))
    monkeypatch.setattr(entrypoint, "_remove_pending_uninstall", lambda release, *, namespace: None)
    monkeypatch.setattr(Helm, "build_dependencies", lambda chart: None)
    monkeypatch.setattr(Helm, "upgrade", staticmethod(fake_upgrade))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)

    KubernetesCommandBackend(_config(run_id, deploy_component)).execute_train(
        train_args=f"--train-backend fsdp {train_args}", num_gpus_per_node=8, megatron_model_type=None
    )
    assert len(recorded) == 1
    return recorded[0]


def values_of(train_argv: list[str]) -> dict[str, Any]:
    return build_values(
        [_router()],
        LaunchPlan(
            run_id=RUN_ID,
            state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
            release="miles-run-260101-000000-000",
            namespace="rl",
            orchestrator_command=["python", "/repo/train.py", *train_argv],
            worker_argv=train_argv,
        ),
    ).as_values()


class TestExecuteTrainTellsThePodsItsBackend:
    def test_the_train_argv_the_launcher_receives_names_kubernetes(self, monkeypatch: pytest.MonkeyPatch):
        """This argv is the only thing the pods are told about the run, so the backend has to be in it."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        assert declared_cluster_backends(argv) == ["kubernetes"]

    def test_the_orchestrator_command_and_the_worker_argv_both_carry_it(self, monkeypatch: pytest.MonkeyPatch):
        """The orchestrator and its workers have to agree, and each reads its own copy of the argv."""
        run = values_of(launch_argv(monkeypatch, train_args="--rollout-num-gpus 8"))["run"]

        assert declared_cluster_backends(run["orchestrator"]["command"]) == ["kubernetes"]
        assert declared_cluster_backends(run["staticWorkers"][0]["command"]) == []

    def test_a_user_supplied_agreeing_flag_is_not_repeated(self, monkeypatch: pytest.MonkeyPatch):
        """A run relaunched from a recorded command line already carries the flag the launcher would add."""
        argv = launch_argv(monkeypatch, train_args="--cluster-backend kubernetes --rollout-num-gpus 8")

        assert argv.count("--cluster-backend") == 1

    def test_a_user_supplied_conflicting_flag_stops_the_launch(self, monkeypatch: pytest.MonkeyPatch):
        """Launching a ray-flagged run onto kubernetes would install pods nothing ever drives."""
        with pytest.raises(AssertionError, match="ray"):
            launch_argv(monkeypatch, train_args="--cluster-backend ray")

    def test_a_user_supplied_agreeing_equals_form_is_not_repeated_either(self, monkeypatch: pytest.MonkeyPatch):
        """`--flag=value` is one token, so a search for the space form would miss it and add a second flag."""
        argv = launch_argv(monkeypatch, train_args="--cluster-backend=kubernetes --rollout-num-gpus 8")

        assert declared_cluster_backends(argv) == ["kubernetes"]
        assert "--cluster-backend=kubernetes" in argv

    def test_the_equals_form_of_a_conflicting_flag_stops_the_launch_too(self, monkeypatch: pytest.MonkeyPatch):
        """The conflict check has to see the same tokens argparse will."""
        with pytest.raises(AssertionError, match="ray"):
            launch_argv(monkeypatch, train_args="--cluster-backend=ray")

    def test_does_not_mistake_another_flags_value_for_a_declaration(self, monkeypatch: pytest.MonkeyPatch):
        """A substring search would find the flag inside `--data-path=--cluster-backend ray` and refuse the launch."""
        argv = launch_argv(monkeypatch, train_args="--data-path='--cluster-backend ray'")

        assert declared_cluster_backends(argv) == ["kubernetes"]
        assert "--data-path=--cluster-backend ray" in argv

    def test_a_trailing_flag_that_names_nothing_stops_the_launch(self, monkeypatch: pytest.MonkeyPatch):
        """argparse would fail on this argv inside the pod, where the failure is much harder to read."""
        with pytest.raises(AssertionError, match="last argument"):
            launch_argv(monkeypatch, train_args="--rollout-num-gpus 8 --cluster-backend")


class TestARunIsNamedBeforeAnythingIsInstalled:
    def test_a_run_id_that_cannot_name_a_kubernetes_object_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        """helm fails on it long after the launcher could have said which value was wrong."""
        with pytest.raises(AssertionError, match="names every object"):
            launch_argv(monkeypatch, train_args="--rollout-num-gpus 8", run_id="Qwen3_4B")


class TestEveryPodOfARunSharesOneRunUuid:
    def test_the_launcher_stamps_one_uuid_on_every_pod(self, monkeypatch: pytest.MonkeyPatch):
        """Left unset, every pod mints its own uuid at parse time and nothing about the run can be joined up."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        [stamped] = ArgvManipulator.get(argv, "--run-uuid")
        assert len(stamped) == RUN_UUID_LENGTH

    def test_two_launches_of_one_run_id_are_two_trainings(self, monkeypatch: pytest.MonkeyPatch):
        """A run id is reused to resume, so deriving the uuid from it would let a stale deployment pass the handshake."""
        first = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")
        second = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        assert ArgvManipulator.get(first, "--run-uuid") != ArgvManipulator.get(second, "--run-uuid")

    def test_a_uuid_the_train_args_already_carry_is_left_alone(self, monkeypatch: pytest.MonkeyPatch):
        """Relaunching from a recorded command line must keep the uuid its artefacts were written under."""
        argv = launch_argv(monkeypatch, train_args="--run-uuid 0123456789abcdef --rollout-num-gpus 8")

        assert ArgvManipulator.get(argv, "--run-uuid") == ["0123456789abcdef"]


class TestThePodDispatchesOnThatFlag:
    def test_that_argv_selects_the_kubernetes_branch_in_the_pod(self, monkeypatch: pytest.MonkeyPatch):
        """The flag is only worth adding if the in-pod dispatch reads it and skips the ray worker manager."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")
        declared = declared_cluster_backends(argv)

        stub = stub_kubernetes_capability(monkeypatch)

        args = SimpleNamespace(cluster_backend=declared[0], num_gpus_per_node=8)
        assert wiring.get_backend_capability(args) is stub.capability
        assert stub.specs_computed_from == [args]
        assert ClusterBackend(declared[0]) is ClusterBackend.KUBERNETES


class TestExecuteTrainTellsThePodsWhichPartOfTheRunTheyAre:
    def test_a_whole_run_is_told_so_too(self, monkeypatch: pytest.MonkeyPatch):
        """Every launch says which part it is, so a pod never has to fall back on a default nobody wrote down."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        assert declared_deploy_components(argv) == ["all"]

    def test_the_train_argv_names_the_part_the_launch_deploys(self, monkeypatch: pytest.MonkeyPatch):
        """The pods filter the run's spec table by this flag, so it is what makes them a subset of the run."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8", deploy_component=DeployComponent.TRAINER)

        assert declared_deploy_components(argv) == ["trainer"]

    def test_the_release_is_named_after_the_part_it_installs(self, monkeypatch: pytest.MonkeyPatch):
        """Two parts of one run id share a release name unless the part is in it, and would uninstall each other."""
        releases: list[str] = []

        launch_argv(
            monkeypatch,
            train_args="--rollout-num-gpus 8",
            deploy_component=DeployComponent.TRAINER,
            recorded_releases=releases,
        )

        assert releases == [f"miles-run-{RUN_ID}-trainer"]

    def test_every_object_of_a_split_release_is_named_after_that_release(self, monkeypatch: pytest.MonkeyPatch):
        """The chart computes no names, so an object named after the whole run would collide with the whole run."""
        run_id = RUN_ID_SHORT_ENOUGH_FOR_FULL_OBJECT_NAMES
        releases: list[str] = []
        argv = launch_argv(
            monkeypatch,
            train_args="--rollout-num-gpus 8",
            run_id=run_id,
            deploy_component=DeployComponent.TRAINER,
            recorded_releases=releases,
        )

        run = _values_of_release(argv, run_id=run_id, release=releases[0])["run"]

        assert run["objectNames"]["orchestrator"] == f"miles-run-{run_id}-trainer-orchestrator"
        assert run["staticWorkers"][0]["objectName"] == f"miles-run-{run_id}-trainer-inference-router-0"

    def test_a_user_supplied_agreeing_flag_is_appended_over_rather_than_detected(self, monkeypatch):
        """A relaunch from a recorded command line repeats the flag, and the last one argparse reads still wins."""
        argv = launch_argv(
            monkeypatch,
            train_args="--deploy-component trainer --rollout-num-gpus 8",
            deploy_component=DeployComponent.TRAINER,
        )

        assert declared_deploy_components(argv) == ["trainer", "trainer"]

    def test_a_user_supplied_conflicting_flag_stops_the_launch(self, monkeypatch: pytest.MonkeyPatch):
        """Everything this launch installs is named after its own part, so pods told another part are orphans."""
        with pytest.raises(AssertionError, match="deploy-component"):
            launch_argv(monkeypatch, train_args="--deploy-component trainer --rollout-num-gpus 8")


class TestApiServerHost:
    def test_a_whole_run_answers_on_its_own_orchestrator(self):
        """The api server runs beside the orchestration script, which is a pod of the run's only release."""
        host = KubernetesCommandBackend(_config()).api_server_host()

        assert host == f"{_release()}-orchestrator.{NAMESPACE}.svc.cluster.local"

    @pytest.mark.parametrize("component", [DeployComponent.PRIMARY, DeployComponent.TRAINER])
    def test_no_deployment_of_a_split_run_has_an_api_server_to_name(self, component):
        """A split run is refused an api server, so any host answered here would only ever time out."""
        backend = KubernetesCommandBackend(_config(deploy_component=component))

        with pytest.raises(AssertionError, match="--api-server-port 0"):
            backend.api_server_host()


def _values_of_release(train_argv: list[str], *, run_id: str, release: str) -> dict[str, Any]:
    return build_values(
        [_router()],
        LaunchPlan(
            run_id=run_id,
            state_file="",
            release=release,
            namespace=NAMESPACE,
            orchestrator_command=[],
            worker_argv=train_argv,
        ),
    ).as_values()


_INSTALLED_MANIFEST = """
kind: StatefulSet
metadata:
  name: miles-run-260101-000000-000-all-orchestrator
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: orchestrator
          command: ["python", "train.py", "--run-uuid", "aaaabbbbccccdddd"]
"""


class TestTheRunUuidALaunchStamps:
    def test_a_split_launch_has_to_be_given_one(self):
        """The parts of a run are joined by nothing else, so a launcher inventing one would split the run in two."""
        config = _config(deploy_component=DeployComponent.TRAINER)
        config.run_uuid = None

        with pytest.raises(AssertionError, match="--run-uuid"):
            entrypoint._resolve_run_uuid(config, installed_manifest=None, release=_release(DeployComponent.TRAINER))

    def test_a_split_launch_keeps_the_one_it_was_given(self):
        """It is the value the layer deploying every part chose, and changing it would fail the handshake."""
        config = _config(deploy_component=DeployComponent.TRAINER)

        assert (
            entrypoint._resolve_run_uuid(config, installed_manifest=None, release=_release(DeployComponent.TRAINER))
            == SPLIT_RUN_UUID
        )

    def test_a_first_install_mints_one(self):
        """A single deployment is the whole run, so nothing outside it has to agree on the value."""
        stamped = entrypoint._resolve_run_uuid(_config(), installed_manifest=None, release=_release())

        assert len(stamped) == RUN_UUID_LENGTH

    def test_an_upgrade_in_place_keeps_the_uuid_the_pods_already_carry(self):
        """Resizing is the same training, and a fresh uuid would change every pod argv and trip the upgrade guard."""
        installed = Manifest.parse(_INSTALLED_MANIFEST, namespace=NAMESPACE)

        assert (
            entrypoint._resolve_run_uuid(_config(), installed_manifest=installed, release=_release())
            == "aaaabbbbccccdddd"
        )
