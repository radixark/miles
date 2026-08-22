from collections.abc import Sequence
from typing import Annotated

import typer
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.deploy.conftest_deploy.common.utils import assert_the_cluster_can_deploy_runs
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import (
    assert_the_take_overs_replaced_only_the_script,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterObserver, observing_the_cluster
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_checkpoint_dir, compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import HOT_RESTART_FORM_NAME, HotRestartFaultForm
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import MetricThresholdOption, NumRolloutOption, SeedOption
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ACTOR_CELL_TYPE, CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import Event, InjectionEvent
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_num_successful_injections_of_form
from tests.e2e.ft.conftest_ft.scenario_realistic_gsm8k import (
    DEFAULT_METRIC_THRESHOLD,
    DEFAULT_NUM_ROLLOUT,
    DEFAULT_SEED,
    Gsm8kRun,
    run_realistic_gsm8k,
)

from miles.utils.external_utils import command_utils
from miles.utils.misc import MutableBox

app: typer.Typer = typer.Typer()

TEST_NAME: str = "hot_restart_realistic_gsm8k"
SAVE_INTERVAL: int = 3
MIN_HOT_RESTARTS: int = 1
MAX_REDONE_STEPS_PER_TAKE_OVER: int = SAVE_INTERVAL + 1
DEFAULT_HOT_RESTART_INTERVAL_SECONDS: float = 600.0

HotRestartIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between take-overs of the orchestration script")
]


@app.command(name="run")
def run_ci(
    seed: SeedOption = DEFAULT_SEED,
    num_rollout: NumRolloutOption = DEFAULT_NUM_ROLLOUT,
    metric_threshold: MetricThresholdOption = DEFAULT_METRIC_THRESHOLD,
    hot_restart_interval_seconds: HotRestartIntervalSecondsOption = DEFAULT_HOT_RESTART_INTERVAL_SECONDS,
) -> None:
    config = command_utils.default_config()
    assert_the_cluster_can_deploy_runs(config)

    observer = ClusterObserver(
        release=compute_release_of_config(config), namespace=config.namespace, trainer_id=DEFAULT_TRAINER_ID
    )
    hot_restart_form: MutableBox[HotRestartFaultForm | None] = MutableBox(value=None)

    def create_forms(run: Gsm8kRun) -> CellFaultForms:
        forms = create_hot_restart_forms(run)
        assert hot_restart_form.value is None, (
            "the run's fault forms were built twice, so the form this soak reads at the end is not the one the "
            "second run was injected with"
        )
        [hot_restart_form.value] = forms[ACTOR_CELL_TYPE]
        return forms

    with observing_the_cluster(observer):
        outcome = run_realistic_gsm8k(
            config=config,
            test_name=TEST_NAME,
            seed=seed,
            num_rollout=num_rollout,
            metric_threshold=metric_threshold,
            fully_async=False,
            mean_interval_seconds_of_cell_type={ACTOR_CELL_TYPE: hot_restart_interval_seconds},
            create_forms=create_forms,
            extra_train_args=build_checkpoint_args(resolve_dump_dir(TEST_NAME)),
        )

    form = hot_restart_form.value
    assert form is not None, "no fault form was ever built for this run, so nothing here was ever taken over"

    form.join_relaunches()
    form.assert_every_take_over_installed_cleanly()
    assert_no_take_over_attempt_failed(outcome.injector.event_log.events)

    evidence = HotRestartEvidence(
        records=form.records,
        snapshots=tuple(observer.snapshots),
        release=observer.release,
        observation_attempts=observer.attempts,
        observation_failures=observer.failures,
    )
    evidence.write(dump_dir=outcome.run.dump_dir)
    assert_the_take_overs_replaced_only_the_script(
        evidence,
        num_restarts=compute_num_successful_injections_of_form(
            outcome.injector.event_log.events, form_name=HOT_RESTART_FORM_NAME
        ),
        minimum_restarts=MIN_HOT_RESTARTS,
    )
    assert_no_take_over_threw_away_more_than_a_save_interval(evidence.records)

    print(f"Hot restart realistic gsm8k test PASSED (seed={seed}, rollouts={num_rollout})")


def assert_no_take_over_attempt_failed(events: list[Event]) -> None:
    failed = [
        one
        for one in events
        if isinstance(one, InjectionEvent) and one.form_name == HOT_RESTART_FORM_NAME and not one.succeeded
    ]

    assert not failed, (
        f"{len(failed)} take-over attempt(s) failed: {failed}. Every draw of this form fires, so a failure here is "
        f"a relaunch the cluster refused or one that never reached the run, not a draw that was declined"
    )


def assert_no_take_over_threw_away_more_than_a_save_interval(records: Sequence[HotRestartRecord]) -> None:
    for record in records:
        resumed_from = -1 if record.saved_iteration_at_trigger is None else record.saved_iteration_at_trigger
        redone = record.frozen_rollout_id - resumed_from

        assert 0 <= redone <= MAX_REDONE_STEPS_PER_TAKE_OVER, (
            f"take-over {record.index} was drawn against a run standing at step {record.frozen_rollout_id} holding "
            f"iteration {record.saved_iteration_at_trigger}, so it threw away {redone} step(s); a take-over resumes "
            f"from the last checkpoint and a run saving every {SAVE_INTERVAL} step(s) cannot owe more than "
            f"{MAX_REDONE_STEPS_PER_TAKE_OVER}"
        )


def create_hot_restart_forms(run: Gsm8kRun) -> CellFaultForms:
    form = HotRestartFaultForm(
        launch=run.launch,
        config=run.config,
        checkpoint_dir=compute_checkpoint_dir(run.dump_dir),
        events_dir=run.events_dir,
    )
    return {ACTOR_CELL_TYPE: [form]}


def build_checkpoint_args(dump_dir: str) -> str:
    checkpoint_dir = compute_checkpoint_dir(dump_dir)
    return f"--save {checkpoint_dir} --load {checkpoint_dir} --save-interval {SAVE_INTERVAL} "


if __name__ == "__main__":
    app()
