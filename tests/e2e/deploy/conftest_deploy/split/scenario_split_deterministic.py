import dataclasses
from collections.abc import Callable

import typer
from examples.infra_features.split_deployment.run_qwen3_0_6b_split import (
    ScriptArgs,
    build_deployment_train_args,
    build_train_args,
    compute_deployment_identities,
)
from tests.e2e.deploy.conftest_deploy.common.example_args import (
    assert_example_parallelism_matches,
    build_deterministic_test_args,
    build_script_args,
    without_weight_decay,
)
from tests.e2e.deploy.conftest_deploy.common.utils import compare_deterministic_sides, run_on_cluster
from tests.e2e.deploy.conftest_deploy.split.split_deployment import RunDeployment, create_split_run_side
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, RunSideRequest, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.execution import DATA_DIR, MODEL_DIR
from tests.e2e.ft.conftest_ft.modes import DENSE_MODEL_HF_REPO, DENSE_MODEL_NAME, DENSE_MODEL_TYPE, FTTestMode

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import get_mooncake_object_store_args

TEST_NAME: str = "split_deterministic"
NUM_ROLLOUTS: int = 3
MIN_TRAINED_ROLLOUTS: int = 2
ROLLOUT_NUM_GPUS_FLAG: str = "--rollout-num-gpus"
ROLLOUT_NUM_GPUS_PER_ENGINE_FLAG: str = "--rollout-num-gpus-per-engine"

_MODE: FTTestMode = FTTestMode(
    model_name=DENSE_MODEL_NAME,
    model_hf_repo=DENSE_MODEL_HF_REPO,
    megatron_model_type=DENSE_MODEL_TYPE,
    num_cells=2,
    train_gpus_per_node=4,
    rollout_num_engines=2,
    rollout_gpus_per_engine=1,
    parallel_args="--context-parallel-size 2",
)


def _build_script_args(
    mode: FTTestMode, dump_dir: str, enable_dumper: bool, config: ExecuteTrainConfig | None = None
) -> ScriptArgs:
    assert mode.has_real_rollout, (
        f"{TEST_NAME} deploys the engines of a run as releases of their own, and mode {mode.model_name} has no "
        f"engines to deploy"
    )
    assert not mode.colocate, (
        f"{TEST_NAME} deploys the trainer and the engines separately, and mode {mode.model_name} colocates them "
        f"on shared gpus"
    )

    return build_script_args(
        config if config is not None else command_utils.default_config(),
        script_args_class=ScriptArgs,
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_rollout=NUM_ROLLOUTS,
        actor_num_gpus=mode.train_gpus_per_node,
        num_engines=mode.rollout_num_engines,
        gpus_per_engine=mode.rollout_gpus_per_engine,
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        extra_args=build_deterministic_test_args(mode=mode, dump_dir=dump_dir, enable_dumper=enable_dumper),
    )


def _build_args(
    mode: FTTestMode, dump_dir: str, enable_dumper: bool = True, config: ExecuteTrainConfig | None = None
) -> str:
    args = _build_script_args(mode, dump_dir, enable_dumper, config)
    train_args = without_weight_decay(build_train_args(args, rollout_num_gpus=mode.total_rollout_gpus))

    assert_example_parallelism_matches(mode, train_args=train_args)
    return train_args


def _build_baseline_args(
    mode: FTTestMode, dump_dir: str, enable_dumper: bool = True, config: ExecuteTrainConfig | None = None
) -> str:
    return _build_args(mode, dump_dir, enable_dumper, config) + get_mooncake_object_store_args()


def _build_deployments(request: RunSideRequest) -> list[RunDeployment]:
    args = _build_script_args(request.mode, request.dump_dir, request.enable_dumper, request.config)

    return [
        RunDeployment(
            deploy_component=deploy_component,
            deploy_instance_id=deploy_instance_id,
            train_args=without_weight_decay(
                build_deployment_train_args(
                    dataclasses.replace(args, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id)
                )
            ),
        )
        for deploy_component, deploy_instance_id in compute_deployment_identities(args)
    ]


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    compare_deterministic_sides(
        baseline_dir=f"{dump_dir}/{BASELINE_SIDE}",
        target_dir=f"{dump_dir}/{TARGET_SIDE}",
        expected_engine_count=mode.rollout_num_engines,
        min_trained_rollouts=MIN_TRAINED_ROLLOUTS,
    )

    print("Split deployment deterministic comparison test PASSED")


# ================================= app wiring =================================


def _create_app_and_run_ci() -> tuple[typer.Typer, Callable[[], None]]:
    app, run_ci = create_comparison_app_and_run_ci(
        test_name=TEST_NAME,
        build_baseline_args=_build_baseline_args,
        build_target_args=_build_args,
        compare_fn=_compare,
        run_side=create_split_run_side(build_baseline_args=_build_baseline_args, build_deployments=_build_deployments),
        resolve_mode_fn=lambda _name: _MODE,
    )
    return app, run_on_cluster(run_ci)


app, run_ci = _create_app_and_run_ci()

if __name__ == "__main__":
    app()
