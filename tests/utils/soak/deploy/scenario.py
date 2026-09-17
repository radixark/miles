import asyncio
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.utils.soak.checks.ft import assert_healing
from tests.utils.soak.checks.weights import assert_published_weight_checksums, assert_weight_checksum_history
from tests.utils.soak.cli_options import NumRolloutOption, SeedOption
from tests.utils.soak.deploy.assert_workloads import (
    _compute_workloads_with_changed_template,
    _compute_workloads_with_replaced_pods,
    assert_take_overs_replaced_only_script,
)
from tests.utils.soak.deploy.cluster_observer import ClusterObserver, ClusterSnapshot, compute_hot_restart_workloads
from tests.utils.soak.deploy.evidence import (
    HotRestartEvidence,
    HotRestartRecord,
    read_discarded_event_dirs,
    read_step_events,
)
from tests.utils.soak.deploy.fault_form import HOT_RESTART_FORM_NAME
from tests.utils.soak.deploy.soak_form import SoakActionFormHotRestart
from tests.utils.soak.deploy.soak_observer import HotRestartSoakObserver
from tests.utils.soak.deploy.soak_session import execute_hot_restart_session
from tests.utils.soak.deploy.utils import compute_checkpoint_dir, compute_release_of_config
from tests.utils.soak.fault_forms import CellFaultForms, create_cell_fault_forms
from tests.utils.soak.recipes.gsm8k import DEFAULT_NUM_ROLLOUT, DEFAULT_SEED, Gsm8kRun, run_realistic_gsm8k
from tests.utils.soak.recipes.gsm8k_launcher import Gsm8kLaunchSpec
from tests.utils.soak.state import (
    SoakActionAppliedEvent,
    SoakActionResultEvent,
    SoakAdmissionClosedEvent,
    SoakDeploymentTarget,
    SoakEvent,
    SoakObservation,
    event_source,
)
from tests.utils.soak.views import project_actions

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

app: typer.Typer = typer.Typer()

TEST_NAME: str = "hot_restart_realistic_gsm8k"
SAVE_INTERVAL: int = 3
MIN_HOT_RESTARTS: int = 2
MAX_REDONE_STEPS_PER_TAKE_OVER: int = SAVE_INTERVAL + 1
DEFAULT_HOT_RESTART_INTERVAL_SECONDS: float = 600.0
TERMINAL_QUIESCENCE_ROLLOUTS: int = 15
_HOT_RESTART_TARGET_TYPE: str = "deployment"

HotRestartIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between take-overs of the orchestration script")
]


@app.command(name="run")
def run_ci(
    seed: SeedOption = DEFAULT_SEED,
    num_rollout: NumRolloutOption = DEFAULT_NUM_ROLLOUT,
    hot_restart_interval_seconds: HotRestartIntervalSecondsOption = DEFAULT_HOT_RESTART_INTERVAL_SECONDS,
    mix_ft: Annotated[bool, typer.Option(help="Mix trainer and rollout faults with deployment takeovers")] = False,
) -> None:
    config = command_utils.default_config()
    assert (
        config.cluster_backend is ClusterBackend.KUBERNETES and config.namespace
    ), "Hot restart needs Kubernetes and a namespace"

    max_allowed_rollout_id = num_rollout - TERMINAL_QUIESCENCE_ROLLOUTS - 1

    def create_forms(run: Gsm8kRun) -> CellFaultForms:
        forms = create_hot_restart_forms(run, max_allowed_rollout_id=max_allowed_rollout_id)
        if mix_ft:
            forms.update(create_cell_fault_forms(base_url=run.base_url, config=run.config))
        return forms

    intervals = {_HOT_RESTART_TARGET_TYPE: hot_restart_interval_seconds}
    if mix_ft:
        intervals.update(actor=120.0, rollout=240.0)

    outcome = asyncio.run(
        run_realistic_gsm8k(
            config=config,
            test_name=f"{TEST_NAME}_mixed" if mix_ft else TEST_NAME,
            seed=seed,
            num_rollout=num_rollout,
            mean_interval_seconds_of_cell_type=intervals,
            create_forms=create_forms,
            create_observer=_create_observer,
            execute_session=execute_hot_restart_session,
            build_extra_train_args=lambda dump_dir: _build_train_args(dump_dir, wandb_run_id=config.run_id),
            enable_fault_tolerance=mix_ft,
        )
    )

    events = outcome.injector.event_log.events
    assert_no_take_over_attempt_failed(events)
    _assert_checkpoints_advanced_between_takeovers(events)

    evidence = _project_evidence(events=events, release=compute_release_of_config(config), namespace=config.namespace)
    evidence.write(dump_dir=str(outcome.run.evidence_dir))
    if mix_ft:
        _assert_mixed_takeover_windows(events=events, forms=outcome.injector.cell_fault_forms)
        assert_healing(
            ("train", "rollout"),
            events=events,
            forms=outcome.injector.cell_fault_forms,
            event_dir=outcome.run.events_dir,
            context="mixed FT/deployment soak",
        )
    else:
        assert_take_overs_replaced_only_script(
            evidence,
            num_restarts=len(evidence.records),
            minimum_restarts=MIN_HOT_RESTARTS,
        )
    assert_take_over_loss_within_save_interval(evidence.records)
    source = event_source(events, name="training_events", fallback=outcome.run.events_dir)
    closures = [event.timestamp for event in events if isinstance(event, SoakAdmissionClosedEvent)]
    assert len(closures) == 1, "Hot restart checksum audit requires one closed admission boundary"
    cutoff = max([*closures, *(event.timestamp for event in events if isinstance(event, SoakActionAppliedEvent))])
    assert_published_weight_checksums(read_events(source), publication_since=cutoff, minimum_publications=2)
    _assert_archived_weight_checksum_history(source)
    assert_take_overs_resumed_within_save_interval(str(source.parent), records=evidence.records)

    print(f"Hot restart realistic gsm8k test PASSED (seed={seed}, rollouts={num_rollout})")


def _assert_archived_weight_checksum_history(source: Path) -> None:
    checksums: dict[str, InferenceEngineWeightChecksumEvent] = {}
    for directory in [*read_discarded_event_dirs(str(source.parent)), source]:
        events = read_events(directory)
        assert_weight_checksum_history(events)
        for event in events:
            if isinstance(event, InferenceEngineWeightChecksumEvent):
                checksums[event.model_dump_json()] = event
    assert_weight_checksum_history(list(checksums.values()))


def _assert_mixed_takeover_windows(*, events: list[SoakEvent], forms: CellFaultForms) -> None:
    actions = project_actions(events)
    form = next(form for form in forms["deployment"] if form.name == HOT_RESTART_FORM_NAME)
    checked = 0
    for action in actions.values():
        target = action.requested.request.target
        if not isinstance(target, SoakDeploymentTarget):
            continue
        start = events.index(action.requested)
        before = next(
            event
            for event in reversed(events[:start])
            if isinstance(event, SoakObservation) and "hot_restart_cluster" in event.details and not event.errors
        )
        end = next(
            (
                index
                for index in range(start + 1, len(events))
                if isinstance(events[index], SoakObservation)
                and form.is_recovered(action=action, events=events[: index + 1])
            ),
            None,
        )
        assert end is not None, "Takeover never recovered before the run ended"
        snapshots = [
            ClusterSnapshot.model_validate(event.details["hot_restart_cluster"])
            for event in [before, *events[start : end + 1]]
            if isinstance(event, SoakObservation) and "hot_restart_cluster" in event.details
        ]
        snapshots = [snapshot for snapshot in snapshots if snapshot.describes_whole_release]
        assert len(snapshots) >= 2, "Takeover has no complete before/after snapshots"
        expected = compute_hot_restart_workloads(target.release)
        assert (
            set(_compute_workloads_with_replaced_pods(snapshots)) == expected
        ), "Takeover replaced non-orchestration pods"
        assert (
            _compute_workloads_with_changed_template(snapshots) == expected
        ), "Takeover changed non-orchestration templates"
        before_uuid = snapshots[0].trainer_boot_uuid
        assert (
            before_uuid and snapshots[-1].trainer_boot_uuid == before_uuid
        ), "Takeover rebooted the trainer controller"
        assert all(snapshot.trainer_boot_uuid in {None, before_uuid} for snapshot in snapshots)
        checked += 1
    assert checked >= MIN_HOT_RESTARTS, "Mixed soak did not cover repeated deployment takeovers"


def _build_train_args(dump_dir: str, *, wandb_run_id: str) -> str:
    return (
        build_checkpoint_args(dump_dir)
        + f"--wandb-run-id {wandb_run_id} "
        + "--ci-disable-weight-update-checker --save-inference-engine-weight-checksum "
    )


def _assert_checkpoints_advanced_between_takeovers(events: list[SoakEvent]) -> None:
    actions = {
        request_id: action
        for request_id, action in project_actions(events).items()
        if action.requested.request.form_name == HOT_RESTART_FORM_NAME
    }
    previous_saved_iteration = -1
    count = 0
    for event in events:
        if not isinstance(event, SoakActionAppliedEvent) or event.request_id not in actions:
            continue
        target = actions[event.request_id].requested.request.target
        assert isinstance(target, SoakDeploymentTarget)
        assert target.saved_iteration is not None and target.saved_iteration > previous_saved_iteration, (
            f"Takeover {event.request_id} lacks a new checkpoint after the preceding takeover: "
            f"saved={target.saved_iteration}, previous={previous_saved_iteration}"
        )
        after = SoakDeploymentTarget.model_validate(event.evidence["after"])
        previous_saved_iteration = max(
            target.saved_iteration, after.saved_iteration if after.saved_iteration is not None else -1
        )
        count += 1
    assert count >= MIN_HOT_RESTARTS, f"Expected at least {MIN_HOT_RESTARTS} applied takeovers, got {count}"


def assert_no_take_over_attempt_failed(events: list[SoakEvent]) -> None:
    actions = {
        request_id: action
        for request_id, action in project_actions(events).items()
        if action.requested.request.form_name == HOT_RESTART_FORM_NAME
    }
    failed = [
        event
        for event in events
        if isinstance(event, SoakActionResultEvent) and event.request_id in actions and not event.returned
    ]

    assert not failed, (
        f"{len(failed)} take-over attempt(s) failed: {failed}. Every draw of this form fires, so a failure here is "
        f"a relaunch the cluster refused or one that never reached the run, not a draw that was declined"
    )
    assert not (
        missing := {request_id for request_id, action in actions.items() if action.applied is None}
    ), f"Take-over requests never applied: {sorted(missing)}"


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
    form = SoakActionFormHotRestart(
        launch_spec=Gsm8kLaunchSpec(config=run.config, train_args=run.train_args),
        event_log=run.event_log,
        log_dir=run.evidence_dir,
        max_allowed_rollout_id=max_allowed_rollout_id,
    )
    return {_HOT_RESTART_TARGET_TYPE: [form]}


def _create_observer(run: Gsm8kRun) -> HotRestartSoakObserver:
    return HotRestartSoakObserver(
        base_url=run.base_url,
        cell_types=set(),
        namespace=run.config.namespace,
        release=compute_release_of_config(run.config),
        trainer_id=DEFAULT_TRAINER_ID,
        checkpoint_dir=compute_checkpoint_dir(run.dump_dir),
        events_dir=run.events_dir,
    )


def _project_evidence(*, events: list[SoakEvent], release: str, namespace: str) -> HotRestartEvidence:
    observer = ClusterObserver(release=release, namespace=namespace, trainer_id=DEFAULT_TRAINER_ID)
    for event in events:
        if isinstance(event, SoakObservation) and (raw := event.details.get("hot_restart_cluster")) is not None:
            observer.record_snapshot(ClusterSnapshot.model_validate(raw))
    records = tuple(
        HotRestartRecord.model_validate(event.evidence["record"])
        for event in events
        if isinstance(event, SoakActionAppliedEvent) and "record" in event.evidence
    )
    return HotRestartEvidence(
        records=records,
        snapshots=tuple(observer.snapshots),
        release=release,
        observation_attempts=observer.attempts,
        observation_failures=observer.failures,
    )


def build_checkpoint_args(dump_dir: str) -> str:
    checkpoint_dir = compute_checkpoint_dir(dump_dir)
    return f"--save {checkpoint_dir} --load {checkpoint_dir} --save-interval {SAVE_INTERVAL} "


if __name__ == "__main__":
    app()
