import asyncio
from typing import Annotated

import typer
from tests.e2e.ft.conftest_ft.cli_options import NumRolloutOption, SeedOption
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.utils import compute_release_of_config
from tests.utils.soak.core.views import event_source
from tests.utils.soak.deploy.checkers.checkpoint_progress import (
    MIN_HOT_RESTARTS,
    SAVE_INTERVAL,
    assert_checkpoints_advanced_between_takeovers,
    assert_take_over_loss_within_save_interval,
    assert_take_overs_resumed_within_save_interval,
)
from tests.utils.soak.deploy.checkers.evidence import project_hot_restart_evidence
from tests.utils.soak.deploy.checkers.launches import assert_hot_restart_launches_finished
from tests.utils.soak.deploy.checkers.takeover_scope import assert_take_overs_replaced_only_script
from tests.utils.soak.deploy.entrypoint import run_hot_restart_soak
from tests.utils.soak.deploy.types import DEPLOYMENT_TARGET_KIND
from tests.utils.soak.deploy.utils import compute_checkpoint_dir
from tests.utils.soak.recipes.gsm8k import DEFAULT_NUM_ROLLOUT, DEFAULT_SEED, Gsm8kRun, prepare_gsm8k_run

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

app: typer.Typer = typer.Typer()

TEST_NAME: str = "hot_restart_realistic_gsm8k"
DEFAULT_HOT_RESTART_INTERVAL_SECONDS: float = 600.0

HotRestartIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between take-overs of the orchestration script")
]


@app.command(name="run")
def run_ci(
    seed: SeedOption = DEFAULT_SEED,
    num_rollout: NumRolloutOption = DEFAULT_NUM_ROLLOUT,
    hot_restart_interval_seconds: HotRestartIntervalSecondsOption = DEFAULT_HOT_RESTART_INTERVAL_SECONDS,
) -> None:
    config = command_utils.default_config()
    assert (
        config.cluster_backend is ClusterBackend.KUBERNETES and config.namespace
    ), "Hot restart needs Kubernetes and a namespace"

    run = prepare_gsm8k_run(
        config=config,
        test_name=TEST_NAME,
        seed=seed,
        num_rollout=num_rollout,
        build_extra_train_args=lambda dump_dir: _build_train_args(dump_dir, wandb_run_id=config.run_id),
        enable_fault_tolerance=False,
    )
    injector = asyncio.run(
        run_hot_restart_soak(
            run=run,
            runner_config=SoakRunnerConfig(
                seed=seed,
                target_configs={
                    DEPLOYMENT_TARGET_KIND: SoakTargetConfig(
                        expected_count=1, mean_interval_seconds=hot_restart_interval_seconds
                    )
                },
                tail=SoakTailConfig.create(num_rollout=num_rollout),
            ),
            evidence_dir=run.evidence_dir,
        )
    )

    _assert_hot_restarts_healthy(run=run, injector=injector, config=config)

    print(f"Hot restart realistic gsm8k test PASSED (seed={seed}, rollouts={num_rollout})")


def _assert_hot_restarts_healthy(
    *, run: Gsm8kRun, injector: SoakRunner, config: command_utils.ExecuteTrainConfig
) -> None:
    events = injector.event_log.events
    assert_hot_restart_launches_finished(events)
    assert_checkpoints_advanced_between_takeovers(events)

    evidence = project_hot_restart_evidence(events, release=compute_release_of_config(config))
    evidence.write(dump_dir=str(run.evidence_dir))
    assert_take_overs_replaced_only_script(
        evidence,
        num_restarts=len(evidence.records),
        minimum_restarts=MIN_HOT_RESTARTS,
    )
    assert_take_over_loss_within_save_interval(evidence.records)
    source = event_source(events, name="training_events", fallback=run.events_dir)
    assert_take_overs_resumed_within_save_interval(str(source.parent), records=evidence.records)


def _build_train_args(dump_dir: str, *, wandb_run_id: str) -> str:
    return (
        build_checkpoint_args(dump_dir)
        + f"--wandb-run-id {wandb_run_id} "
        + "--ci-disable-weight-update-checker --save-inference-engine-weight-checksum "
    )


def build_checkpoint_args(dump_dir: str) -> str:
    checkpoint_dir = compute_checkpoint_dir(dump_dir)
    return f"--save {checkpoint_dir} --load {checkpoint_dir} --save-interval {SAVE_INTERVAL} "


if __name__ == "__main__":
    app()
