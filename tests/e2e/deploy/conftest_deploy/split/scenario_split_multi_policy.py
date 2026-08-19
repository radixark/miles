import dataclasses
import os

import typer

from examples.infra_features.split_deployment.run_solver_verifier_gsm8k_split import (
    build_deployment_train_args,
    compute_deployment_identities,
)
from examples.multi_policy.run_solver_verifier_gsm8k import (
    TRAIN_EXTRA_ENV_VARS,
    ScriptArgs,
    compute_events_dir,
    launch_train,
    prepare,
)
from tests.e2e.deploy.conftest_deploy.common.utils import assert_cluster_can_deploy_runs
from tests.e2e.deploy.conftest_deploy.split.split_deployment import RunDeployment, run_split_training_into

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.run_uuid import generate_run_uuid

TEST_NAME: str = "split_multi_policy"


def _run(args: ScriptArgs) -> None:
    prepare(args)

    config = dataclasses.replace(args, run_uuid=generate_run_uuid())
    run_split_training_into(
        deployments=_build_deployments(config),
        launch=launch_train,
        config=config,
        dump_dir=_compute_dump_dir(config),
    )


def _build_deployments(args: ScriptArgs) -> list[RunDeployment]:
    return [
        RunDeployment(
            deploy_component=deploy_component,
            deploy_instance_id=deploy_instance_id,
            train_args=build_deployment_train_args(
                dataclasses.replace(args, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id)
            ),
        )
        for deploy_component, deploy_instance_id in compute_deployment_identities(args)
    ]


def _compute_dump_dir(config: ExecuteTrainConfig) -> str:
    return str(compute_events_dir(config).parent)


# ================================= app wiring =================================


def _run_ci_on_deployable_cluster() -> None:
    config = command_utils.default_config(ScriptArgs)
    assert_cluster_can_deploy_runs(config)
    os.environ.update(TRAIN_EXTRA_ENV_VARS)
    _run(config)


app: typer.Typer = typer.Typer()
app.command(name="run")(_run_ci_on_deployable_cluster)
run_ci = _run_ci_on_deployable_cluster

if __name__ == "__main__":
    app()
