import asyncio
import functools
from typing import Annotated

import typer
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.ft.conftest_ft.cli_options import NumRolloutOption, SeedOption
from tests.utils.soak.core.types import SoakForms, SoakObserver
from tests.utils.soak.core.utils import compute_release_of_config
from tests.utils.soak.core.views import event_source
from tests.utils.soak.deploy.actions.hot_restart import HotRestartForm
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
from tests.utils.soak.deploy.observers import DeploymentObserver
from tests.utils.soak.deploy.session import LauncherChain, execute_hot_restart_session
from tests.utils.soak.deploy.types import DEPLOYMENT_TARGET_KIND
from tests.utils.soak.deploy.utils import compute_checkpoint_dir
from tests.utils.soak.recipes.gsm8k import (
    DEFAULT_NUM_ROLLOUT,
    DEFAULT_SEED,
    Gsm8kOutcome,
    Gsm8kRun,
    run_realistic_gsm8k,
)

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

app: typer.Typer = typer.Typer()

TEST_NAME: str = "hot_restart_realistic_gsm8k"
DEFAULT_HOT_RESTART_INTERVAL_SECONDS: float = 600.0
TERMINAL_QUIESCENCE_ROLLOUTS: int = 15

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

    outcome = asyncio.run(
        _run_soak(
            config=config,
            seed=seed,
            num_rollout=num_rollout,
            hot_restart_interval_seconds=hot_restart_interval_seconds,
        )
    )

    _assert_hot_restarts_healthy(outcome=outcome, config=config)

    print(f"Hot restart realistic gsm8k test PASSED (seed={seed}, rollouts={num_rollout})")


async def _run_soak(
    *,
    config: command_utils.ExecuteTrainConfig,
    seed: int,
    num_rollout: int,
    hot_restart_interval_seconds: float,
) -> Gsm8kOutcome:
    max_allowed_rollout_id = num_rollout - TERMINAL_QUIESCENCE_ROLLOUTS - 1
    chain = LauncherChain()

    def create_forms(run: Gsm8kRun) -> SoakForms:
        return create_hot_restart_forms(run, max_allowed_rollout_id=max_allowed_rollout_id, chain=chain)

    def create_observer(run: Gsm8kRun, forms: SoakForms) -> SoakObserver:
        return DeploymentObserver(
            namespace=run.launch_spec.config.namespace,
            release=compute_release_of_config(run.launch_spec.config),
            trainer_id=DEFAULT_TRAINER_ID,
            checkpoint_dir=compute_checkpoint_dir(run.dump_dir),
            events_dir=run.events_dir,
        )

    return await run_realistic_gsm8k(
        config=config,
        test_name=TEST_NAME,
        seed=seed,
        num_rollout=num_rollout,
        mean_interval_seconds_of_kind={DEPLOYMENT_TARGET_KIND: hot_restart_interval_seconds},
        expected_counts={DEPLOYMENT_TARGET_KIND: 1},
        create_forms=create_forms,
        create_observer=create_observer,
        execute_session=functools.partial(execute_hot_restart_session, chain=chain),
        build_extra_train_args=lambda dump_dir: _build_train_args(dump_dir, wandb_run_id=config.run_id),
        enable_fault_tolerance=False,
    )


def _assert_hot_restarts_healthy(*, outcome: Gsm8kOutcome, config: command_utils.ExecuteTrainConfig) -> None:
    events = outcome.injector.event_log.events
    assert_hot_restart_launches_finished(events)
    assert_checkpoints_advanced_between_takeovers(events)

    evidence = project_hot_restart_evidence(
        events, release=compute_release_of_config(config), namespace=config.namespace
    )
    evidence.write(dump_dir=str(outcome.run.evidence_dir))
    assert_take_overs_replaced_only_script(
        evidence,
        num_restarts=len(evidence.records),
        minimum_restarts=MIN_HOT_RESTARTS,
    )
    assert_take_over_loss_within_save_interval(evidence.records)
    source = event_source(events, name="training_events", fallback=outcome.run.events_dir)
    assert_take_overs_resumed_within_save_interval(str(source.parent), records=evidence.records)


def _build_train_args(dump_dir: str, *, wandb_run_id: str) -> str:
    return (
        build_checkpoint_args(dump_dir)
        + f"--wandb-run-id {wandb_run_id} "
        + "--ci-disable-weight-update-checker --save-inference-engine-weight-checksum "
    )


def create_hot_restart_forms(run: Gsm8kRun, *, max_allowed_rollout_id: int, chain: LauncherChain) -> SoakForms:
    form = HotRestartForm(
        launch_spec=run.launch_spec,
        event_log=run.event_log,
        max_allowed_rollout_id=max_allowed_rollout_id,
        chain=chain,
    )
    return {DEPLOYMENT_TARGET_KIND: [form]}


def build_checkpoint_args(dump_dir: str) -> str:
    checkpoint_dir = compute_checkpoint_dir(dump_dir)
    return f"--save {checkpoint_dir} --load {checkpoint_dir} --save-interval {SAVE_INTERVAL} "


if __name__ == "__main__":
    app()
