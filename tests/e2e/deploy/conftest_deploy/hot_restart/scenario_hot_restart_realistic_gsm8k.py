import asyncio
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import HOT_RESTART_FORM_NAME, HotRestartFaultForm
from tests.e2e.ft.conftest_ft.cli_options import NumRolloutOption, SeedOption
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import Event, InjectionEvent
from tests.utils.deploy.hot_restart.evidence import HotRestartRecord, read_discarded_event_dirs, read_step_events
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.utils import compute_release_of_config
from tests.utils.soak.core.views import event_source
from tests.utils.soak.deploy.checkers import checkpoint_progress
from tests.utils.soak.deploy.checkers.checkpoint_progress import assert_checkpoints_advanced_between_takeovers
from tests.utils.soak.deploy.checkers.evidence import project_hot_restart_evidence
from tests.utils.soak.deploy.checkers.launches import assert_hot_restart_launches_finished
from tests.utils.soak.deploy.checkers.takeover_scope import assert_take_overs_replaced_only_script
from tests.utils.soak.deploy.entrypoint import run_hot_restart_soak
from tests.utils.soak.deploy.types import DEPLOYMENT_TARGET_KIND
from tests.utils.soak.deploy.utils import compute_checkpoint_dir
from tests.utils.soak.recipes.gsm8k import (
    DEFAULT_NUM_ROLLOUT,
    DEFAULT_SEED,
    Gsm8kRun,
    _LegacyGsm8kRun,
    prepare_gsm8k_run,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

app: typer.Typer = typer.Typer()

TEST_NAME: str = "hot_restart_realistic_gsm8k"
SAVE_INTERVAL: int = 3
MIN_HOT_RESTARTS: int = 1
MAX_REDONE_STEPS_PER_TAKE_OVER: int = SAVE_INTERVAL + 1
DEFAULT_HOT_RESTART_INTERVAL_SECONDS: float = 600.0
_HOT_RESTART_CELL_TYPE: str = "hot-restart-virtual-cell"
_VIRTUAL_CELL_NAMES: tuple[str, str] = ("hot-restart-virtual-cell-0", "hot-restart-virtual-cell-1")

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
        minimum_restarts=checkpoint_progress.MIN_HOT_RESTARTS,
    )
    checkpoint_progress.assert_take_over_loss_within_save_interval(evidence.records)
    source = event_source(events, name="training_events", fallback=run.events_dir)
    checkpoint_progress.assert_take_overs_resumed_within_save_interval(str(source.parent), records=evidence.records)


def _build_train_args(dump_dir: str, *, wandb_run_id: str) -> str:
    return (
        build_checkpoint_args(dump_dir)
        + f"--wandb-run-id {wandb_run_id} "
        + "--ci-disable-weight-update-checker --save-inference-engine-weight-checksum "
    )


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


def create_hot_restart_forms(run: _LegacyGsm8kRun, *, max_allowed_rollout_id: int) -> CellFaultForms:
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
