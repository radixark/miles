import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.deploy.conftest_deploy.common.utils import assert_cluster_can_deploy_runs
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_checkpoint import (
    read_discarded_event_dirs,
    read_step_events,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import assert_take_overs_replaced_only_script
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterObserver, observing_cluster
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_checkpoint_dir, compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import HOT_RESTART_FORM_NAME, HotRestartFaultForm
from tests.e2e.ft.conftest_ft.cli_options import MetricThresholdOption, NumRolloutOption, SeedOption
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import Event, InjectionEvent
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_num_successful_injections_of_form
from tests.e2e.ft.conftest_ft.scenario_realistic_gsm8k import (
    DEFAULT_METRIC_THRESHOLD,
    DEFAULT_NUM_ROLLOUT,
    DEFAULT_SEED,
    Gsm8kRun,
    run_realistic_gsm8k,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.misc import MutableBox

app: typer.Typer = typer.Typer()

TEST_NAME: str = "hot_restart_realistic_gsm8k"
SAVE_INTERVAL: int = 3
MIN_HOT_RESTARTS: int = 1
MAX_REDONE_STEPS_PER_TAKE_OVER: int = SAVE_INTERVAL + 1
DEFAULT_HOT_RESTART_INTERVAL_SECONDS: float = 600.0
TERMINAL_QUIESCENCE_ROLLOUTS: int = 15
_HOT_RESTART_CELL_TYPE: str = "hot-restart-virtual-cell"
_VIRTUAL_CELL_NAMES: tuple[str, str] = ("hot-restart-virtual-cell-0", "hot-restart-virtual-cell-1")

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
    assert_cluster_can_deploy_runs(config)

    observer = ClusterObserver(
        release=compute_release_of_config(config), namespace=config.namespace, trainer_id=DEFAULT_TRAINER_ID
    )
    hot_restart_form: MutableBox[HotRestartFaultForm | None] = MutableBox(value=None)
    max_allowed_rollout_id = num_rollout - TERMINAL_QUIESCENCE_ROLLOUTS - 1

    def create_forms(run: Gsm8kRun) -> CellFaultForms:
        forms = create_hot_restart_forms(run, max_allowed_rollout_id=max_allowed_rollout_id)
        assert hot_restart_form.value is None, (
            "the run's fault forms were built twice, so the form this soak reads at the end is not the one the "
            "second run was injected with"
        )
        [hot_restart_form.value] = forms[_HOT_RESTART_CELL_TYPE]
        return forms

    with observing_cluster(observer):
        outcome = run_realistic_gsm8k(
            config=config,
            test_name=TEST_NAME,
            seed=seed,
            num_rollout=num_rollout,
            metric_threshold=metric_threshold,
            fully_async=False,
            mean_interval_seconds_of_cell_type={_HOT_RESTART_CELL_TYPE: hot_restart_interval_seconds},
            create_forms=create_forms,
            get_virtual_cells=lambda: _create_virtual_cells_before(hot_restart_form.value),
            build_extra_train_args=_build_train_args,
            enable_fault_tolerance=False,
        )

    form = hot_restart_form.value
    assert form is not None, "no fault form was ever built for this run, so nothing here was ever taken over"

    form.join_relaunches()
    form.assert_take_overs_installed_cleanly()
    assert_no_take_over_attempt_failed(outcome.injector.event_log.events)

    evidence = HotRestartEvidence(
        records=form.records,
        snapshots=tuple(observer.snapshots),
        release=observer.release,
        observation_attempts=observer.attempts,
        observation_failures=observer.failures,
    )
    evidence.write(dump_dir=outcome.run.dump_dir)
    assert_take_overs_replaced_only_script(
        evidence,
        num_restarts=compute_num_successful_injections_of_form(
            outcome.injector.event_log.events, form_name=HOT_RESTART_FORM_NAME
        ),
        minimum_restarts=MIN_HOT_RESTARTS,
    )
    assert_take_over_loss_within_save_interval(evidence.records)
    assert_take_overs_resumed_within_save_interval(outcome.run.dump_dir, records=evidence.records)

    print(f"Hot restart realistic gsm8k test PASSED (seed={seed}, rollouts={num_rollout})")


def _build_train_args(dump_dir: str) -> str:
    return build_checkpoint_args(dump_dir) + "--ci-disable-weight-update-checker "


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


def assert_take_over_loss_within_save_interval(records: Sequence[HotRestartRecord]) -> None:
    for record in records:
        resumed_from = -1 if record.saved_iteration_at_trigger is None else record.saved_iteration_at_trigger
        redone = record.frozen_rollout_id - resumed_from

        assert 0 <= redone <= MAX_REDONE_STEPS_PER_TAKE_OVER, (
            f"take-over {record.index} was drawn against a run standing at step {record.frozen_rollout_id} holding "
            f"iteration {record.saved_iteration_at_trigger}, so it threw away {redone} step(s); a take-over resumes "
            f"from the last checkpoint and a run saving every {SAVE_INTERVAL} step(s) cannot owe more than "
            f"{MAX_REDONE_STEPS_PER_TAKE_OVER}"
        )


def assert_take_overs_resumed_within_save_interval(dump_dir: str, *, records: Sequence[HotRestartRecord]) -> None:
    logs = _read_replaced_logs(dump_dir, num_take_overs=len(records))

    for record, log, later_log in zip(records, logs[:-1], logs[1:], strict=True):
        frozen_rollout_id = max(log, default=-1)
        survived = sorted(rollout_id for rollout_id, event in log.items() if later_log.get(rollout_id) == event)
        resumed_from = max(survived, default=-1)

        assert survived == list(
            range(resumed_from + 1)
        ), f"take-over {record.index} carried the steps {survived} over, expected {list(range(resumed_from + 1))}"
        redone = frozen_rollout_id - resumed_from
        assert 0 <= redone <= MAX_REDONE_STEPS_PER_TAKE_OVER, (
            f"take-over {record.index} replaced a log that had reached step {frozen_rollout_id} and resumed at "
            f"step {resumed_from}, so it redid {redone} step(s), more than {MAX_REDONE_STEPS_PER_TAKE_OVER}"
        )


def _read_replaced_logs(dump_dir: str, *, num_take_overs: int) -> list[dict[int, str]]:
    discarded_dirs = read_discarded_event_dirs(dump_dir)
    assert len(discarded_dirs) == num_take_overs, (
        f"every take-over rolls the log it replaced aside, but {num_take_overs} of them left "
        f"{[one.name for one in discarded_dirs]} under {dump_dir}"
    )

    rolled_aside_at = [_read_log_rollaside_times(one) for one in discarded_dirs]
    assert (
        sorted(set(rolled_aside_at)) == rolled_aside_at
    ), f"the take-overs under {dump_dir} rolled their logs aside at {rolled_aside_at}, two in the same second"

    replaced = [_read_finished_steps_of_log(one) for one in discarded_dirs]
    return [*replaced, _read_finished_steps_of_log(Path(dump_dir) / EVENTS_DIRNAME)]


def _read_log_rollaside_times(events_dir: Path) -> str:
    matched = re.fullmatch(r"\.trash_(\d{8}_\d{6})_[0-9a-f]+", events_dir.name)
    assert matched is not None, f"{events_dir.name} does not name the moment the log was rolled aside"
    return matched.group(1)


def _read_finished_steps_of_log(events_dir: Path) -> dict[int, str]:
    logged = read_step_events(events_dir)
    repeated = {rollout_id: len(events) for rollout_id, events in logged.items() if len(events) != 1}
    assert not repeated, f"{events_dir} describes the step(s) {repeated} more than once"
    return {rollout_id: events[0] for rollout_id, events in logged.items()}


def create_hot_restart_forms(run: Gsm8kRun, *, max_allowed_rollout_id: int) -> CellFaultForms:
    form = HotRestartFaultForm(
        launch=run.launch,
        config=run.config,
        checkpoint_dir=compute_checkpoint_dir(run.dump_dir),
        events_dir=run.events_dir,
        max_allowed_rollout_id=max_allowed_rollout_id,
    )
    return {_HOT_RESTART_CELL_TYPE: [form]}


def _create_virtual_cells() -> list[dict]:
    return [
        {
            "metadata": {"name": name, "labels": {"miles.io/cell-type": _HOT_RESTART_CELL_TYPE}},
            "status": {"phase": "Running", "conditions": [{"type": "Healthy", "status": "True"}]},
        }
        for name in _VIRTUAL_CELL_NAMES
    ]


def _create_virtual_cells_before(form: HotRestartFaultForm | None) -> list[dict]:
    if form is None or not form.is_within_injection_window():
        return []
    return _create_virtual_cells()


def build_checkpoint_args(dump_dir: str) -> str:
    checkpoint_dir = compute_checkpoint_dir(dump_dir)
    return f"--save {checkpoint_dir} --load {checkpoint_dir} --save-interval {SAVE_INTERVAL} "


if __name__ == "__main__":
    app()
