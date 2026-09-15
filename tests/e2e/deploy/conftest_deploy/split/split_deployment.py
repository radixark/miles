import dataclasses
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass

from tests.e2e.ft.conftest_ft.app import TARGET_SIDE, RunSideFn, RunSideRequest, run_one_release
from tests.e2e.ft.conftest_ft.execution import run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.run_uuid import generate_run_uuid
from miles.utils.workers.types import DeployComponent

logger = logging.getLogger(__name__)


# ============================== deployment types ==============================


@dataclass(frozen=True)
class RunDeployment:
    deploy_component: DeployComponent
    train_args: str
    deploy_instance_id: str | None = None

    def release(self, run_id: str) -> str:
        return ReleaseName(
            run_id=run_id, deploy_component=self.deploy_component, deploy_instance_id=self.deploy_instance_id
        ).serialize()


LaunchDeploymentFn = Callable[[str, command_utils.ExecuteTrainConfig], None]
BuildDeploymentsFn = Callable[[RunSideRequest], list[RunDeployment]]
BuildBaselineArgsFn = Callable[[FTTestMode, str, bool, command_utils.ExecuteTrainConfig], str]


# ============================== run side wiring ===============================


def create_split_run_side(
    *, build_baseline_args: BuildBaselineArgsFn, build_deployments: BuildDeploymentsFn
) -> RunSideFn:
    def run_side(request: RunSideRequest) -> None:
        if request.side != TARGET_SIDE:
            baseline = build_baseline_args(request.mode, request.dump_dir, request.enable_dumper, request.config)
            run_one_release(dataclasses.replace(request, train_args=baseline))
            return

        split = dataclasses.replace(request, config=dataclasses.replace(request.config, run_uuid=generate_run_uuid()))
        run_split_training_into(
            deployments=build_deployments(split),
            launch=create_launch_of_mode(split.mode),
            config=split.config,
            dump_dir=split.dump_dir,
        )

    return run_side


def create_launch_of_mode(mode: FTTestMode) -> LaunchDeploymentFn:
    def launch(train_args: str, config: command_utils.ExecuteTrainConfig) -> None:
        run_training(train_args=train_args, mode=mode, config=config)

    return launch


# ========================= installing the deployments =========================


def run_split_training_into(
    *,
    deployments: list[RunDeployment],
    launch: LaunchDeploymentFn,
    config: command_utils.ExecuteTrainConfig,
    dump_dir: str,
) -> None:
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)

    run_split_training(deployments=deployments, launch=launch, config=config)


def run_split_training(
    *, deployments: list[RunDeployment], launch: LaunchDeploymentFn, config: command_utils.ExecuteTrainConfig
) -> None:
    _assert_deployments_form_one_run(deployments)
    assert config.namespace, (
        "the deployments of one run reach each other by in-cluster release names, and a namespace is half of "
        "every such name"
    )
    assert config.run_uuid is not None, (
        f"the deployments of run {config.run_id} are installed by launches of their own and joined by nothing but "
        f"the run uuid, so whatever installs them all has to name it"
    )

    driver_release = deployments[-1].release(config.run_id)
    installed: list[str] = []
    try:
        for deployment in deployments:
            release = deployment.release(config.run_id)
            installed.append(release)
            logger.info(
                f"Installing {release}, the {deployment.deploy_component.value} deployment of run "
                f"{config.run_id} ({config.run_uuid})"
            )
            launch(
                deployment.train_args,
                dataclasses.replace(
                    config,
                    deploy_component=deployment.deploy_component,
                    deploy_instance_id=deployment.deploy_instance_id,
                ),
            )
            if release != driver_release:
                _assert_release_installed(release=release, namespace=config.namespace)
    finally:
        for release in reversed(installed):
            if release != driver_release:
                _uninstall_whatever_can_be_uninstalled(release=release, namespace=config.namespace)


# ============================= cleanup and checks =============================


def _uninstall_whatever_can_be_uninstalled(*, release: str, namespace: str) -> None:
    try:
        Helm.uninstall(release=release, namespace=namespace)
    except Exception:
        logger.exception(
            f"Uninstalling {release} in namespace {namespace} failed, so it may still hold the gpus of this run; "
            f"the releases installed before it are uninstalled next"
        )


def _assert_deployments_form_one_run(deployments: list[RunDeployment]) -> None:
    unsplit = [one.deploy_component.value for one in deployments if not one.deploy_component.is_split()]
    assert not unsplit, (
        f"these deployments install {unsplit}, which carries a whole run in one release: a run installed that way "
        f"is the unsplit shape, and nothing here would be deployed apart from anything else"
    )

    drivers = [one for one in deployments if one.deploy_component.deploys_orchestration_script()]
    assert len(drivers) == 1, (
        f"exactly one deployment of a run carries the orchestration script, and these carry it {len(drivers)} "
        f"time(s): {[one.deploy_component.value for one in deployments]}"
    )
    assert deployments[-1] is drivers[0], (
        f"installing the orchestration script blocks until the run finishes, so it goes last; here it is "
        f"{deployments.index(drivers[0])} of {len(deployments)}"
    )

    named = [(one.deploy_component, one.deploy_instance_id) for one in deployments]
    assert len(set(named)) == len(named), (
        f"two of these deployments are named the same and would install one release over the other: "
        f"{[(one.value, instance) for one, instance in named]}"
    )

    for deployment in deployments:
        assert deployment.deploy_instance_id is None or deployment.deploy_component.takes_instance_id(), (
            f"a run holds one {deployment.deploy_component.value} deployment, so naming this one "
            f"{deployment.deploy_instance_id!r} tells it apart from nothing"
        )


def _assert_release_installed(*, release: str, namespace: str) -> None:
    assert Helm.get_manifest(release, namespace) is not None, (
        f"installing {release} returned without an error, but helm does not know it in namespace {namespace}: the "
        f"run would wait forever for workers nothing carries"
    )
