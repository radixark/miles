# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import (
    AllowDuringRecoveryOption,
    FullyAsyncOption,
    MaxConcurrentActionsOption,
    MinSurvivorsOption,
    ModeOption,
    NumStepsOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)
from tests.e2e.ft.conftest_ft.execution import (
    get_common_train_args,
    get_ft_args,
    materialize_cyclic_debug_rollout_data,
    prepare,
    run_training,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode, resolve_mode
from tests.utils.soak.checks.ft import assert_healing
from tests.utils.soak.checks.hooks import (
    assert_batch_trainers_recovered,
    assert_hook_effects,
    assert_hook_survivors,
    assert_remote_p2p_failures,
)
from tests.utils.soak.checks.tail import assert_tail_complete
from tests.utils.soak.config import create_policy, create_tail_policy
from tests.utils.soak.entrypoint import API_SERVER_PORT, spawn_fault_injector
from tests.utils.soak.fault_forms import compute_mean_interval_seconds_of_cell_type, create_cell_fault_forms
from tests.utils.soak.hook_fault_form import HookFaultForm
from tests.utils.soak.state import event_source
from tests.utils.soak.teardown import teardown_run
from tests.utils.soak.utils import (
    create_soak_config,
    evidence_directory,
    get_api_server_args,
    get_fully_async_args,
    get_train_script,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import FaultHookEvent, TrainGroupStepEndEvent, WeightUpdateResultEvent
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector import FailureMode

app: typer.Typer = typer.Typer()

TEST_NAME: str = "random_crash"

DEFAULT_SEED: int = 42
DEFAULT_NUM_STEPS: int = 60
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 120.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 240.0


@app.command(name="run")
def run_ci(
    mode: ModeOption,
    seed: SeedOption = DEFAULT_SEED,
    num_steps: NumStepsOption = DEFAULT_NUM_STEPS,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    fully_async: FullyAsyncOption = False,
    allow_during_recovery: AllowDuringRecoveryOption = True,
    min_survivors: MinSurvivorsOption = 1,
    max_concurrent_actions: MaxConcurrentActionsOption = 1,
    precise_all_gather: Annotated[bool, typer.Option()] = False,
    precise_p2p: Annotated[bool, typer.Option()] = False,
    all_p2p_targets: Annotated[bool, typer.Option()] = False,
) -> None:
    """Random failure soak test, for whichever components the mode enables ft on.

    Starts a background thread that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    if all_p2p_targets:
        assert precise_p2p and mode == "kill_rollout__dp2_tp2", "All-target faults require the rollout P2P scenario"
        assert min_survivors == 2, "All-sender-target faults must preserve the other sender's two targets"
    assert not (precise_all_gather and precise_p2p), "Select one precise hook scenario"
    if precise_all_gather:
        assert mode == "kill_train__dp2_tp2", "Precise all-gather requires the real-rollout TP2 mode"
    if precise_p2p:
        assert mode in {
            "kill_train__dp2_tp2",
            "kill_rollout__dp2_tp2",
        }, "Precise P2P requires a disaggregated TP2 mode"
    tail_policy = create_tail_policy(num_rollout=num_steps)
    if fully_async:
        assert_mode_supports_fully_async(ft_mode, mode=mode)

    config = create_soak_config(command_utils.default_config())
    test_name: str = f"{TEST_NAME}_fully_async" if fully_async else TEST_NAME
    if precise_all_gather:
        test_name = f"precise_all_gather{'_fully_async' if fully_async else ''}"
    if precise_p2p:
        test_name = f"precise_p2p{'_all_targets' if all_p2p_targets else ''}{'_fully_async' if fully_async else ''}"
    dump_dir: str = resolve_dump_dir(f"{test_name}_{mode}", run_id=config.run_id)
    print(f"Dump directory: {dump_dir}")
    mean_interval_seconds_of_cell_type: dict[str, float] = compute_mean_interval_seconds_of_cell_type(
        ft_mode.ft_components,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
    )
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection intervals: {mean_interval_seconds_of_cell_type}")
    print(f"FT components: {ft_mode.ft_components}, cluster backend: {config.cluster_backend.value}")
    print(f"Train script: {get_train_script(fully_async=fully_async)}")

    prepare(ft_mode, config=config)

    debug_rollout_data_dir = None if ft_mode.has_real_rollout else materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(
            ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=debug_rollout_data_dir
        )
        + get_ft_args(
            replace(ft_mode, ft_components=("train", "rollout")) if all_p2p_targets else ft_mode,
            api_server_args=get_api_server_args(config),
        )
        + get_fully_async_args(fully_async=fully_async)
        + "--mini-ft-controller-enable "
    )
    if precise_all_gather or precise_p2p:
        train_args += "--update-weight-transfer-mode p2p --train-step-timeout 600 --update-weights-timeout 600 "

    base_url = f"http://{config.create_backend().api_server_host(config)}:{API_SERVER_PORT}"
    evidence_dir = evidence_directory(Path(dump_dir))
    cell_fault_forms = (
        {
            "actor": [
                HookFaultForm(
                    base_url=base_url,
                    failure_mode=failure_mode,
                    hook="trainer_before_all_gather" if precise_all_gather else "trainer_before_weight_send",
                    lifetime_seconds=300,
                )
                for failure_mode in [FailureMode.SIGKILL, FailureMode.DEADLOCK, FailureMode.THREAD_DEADLOCK]
            ]
        }
        if precise_all_gather or (precise_p2p and ft_mode.ft_components == ("train",))
        else create_cell_fault_forms(base_url=base_url, config=config)
    )
    if precise_p2p and ft_mode.ft_components == ("rollout",):
        cell_fault_forms = {
            "rollout": [
                HookFaultForm(
                    base_url=base_url,
                    failure_mode=FailureMode.SIGSTOP if "sigstop" in victim.name else FailureMode.SIGKILL,
                    hook="trainer_before_weight_send",
                    lifetime_seconds=300,
                    victim_form=victim,
                    all_targets=all_p2p_targets,
                )
                for victim in cell_fault_forms["rollout"]
            ]
        }
    injector = spawn_fault_injector(
        tail_policy=tail_policy,
        policy=create_policy(
            expected_cells={
                kind: count
                for kind, count in {"actor": ft_mode.num_cells, "rollout": ft_mode.rollout_num_engines}.items()
                if kind in mean_interval_seconds_of_cell_type
            },
            allow_during_recovery=allow_during_recovery,
            min_survivors=min_survivors,
            max_concurrent_actions=max_concurrent_actions,
        ),
        evidence_path=evidence_dir / "events.jsonl",
        sources={"training_events": Path(dump_dir) / EVENTS_DIRNAME},
        config=config,
        base_url=base_url,
        seed=seed,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        cell_fault_forms=cell_fault_forms,
    )

    try:
        run_training(
            injector=injector,
            train_args=train_args,
            mode=ft_mode,
            dump_dir=dump_dir,
            extra_env_vars={},
            config=config,
            train_script=get_train_script(fully_async=fully_async),
        )
    finally:
        injector.stop_and_join(
            teardown=partial(teardown_run, config=config, event_log=injector.event_log, evidence_dir=evidence_dir)
        )

    assert_tail_complete(injector.event_log.events)
    if precise_all_gather or precise_p2p:
        hook_event_dir = event_source(
            injector.event_log.events, name="training_events", fallback=Path(dump_dir) / EVENTS_DIRNAME
        )
        training_events = read_events(hook_event_dir)
        assert_hook_effects(
            injector.event_log.events,
            hook_events=[event for event in training_events if isinstance(event, FaultHookEvent)],
        )
        if precise_p2p and "rollout" in ft_mode.ft_components:
            matched_request_ids = assert_remote_p2p_failures(
                injector.event_log.events,
                hook_events=[event for event in training_events if isinstance(event, FaultHookEvent)],
                update_events=[event for event in training_events if isinstance(event, WeightUpdateResultEvent)],
                require_all_targets_failed=all_p2p_targets,
            )
            if all_p2p_targets:
                assert_batch_trainers_recovered(
                    injector.event_log.events,
                    steps=[event for event in training_events if isinstance(event, TrainGroupStepEndEvent)],
                    expected_trainers=ft_mode.num_cells,
                    matched_request_ids=matched_request_ids,
                )
        if "train" in ft_mode.ft_components:
            assert_hook_survivors(
                injector.event_log.events,
                steps=[event for event in training_events if isinstance(event, TrainGroupStepEndEvent)],
            )
    assert_healing(
        ft_mode.ft_components,
        injector=injector,
        event_dir=Path(dump_dir) / EVENTS_DIRNAME,
        context=f"{test_name} {mode}",
    )

    print(f"Random failure soak test PASSED ({test_name}, mode={mode}, seed={seed}, steps={num_steps})")


def assert_mode_supports_fully_async(ft_mode: FTTestMode, *, mode: str) -> None:
    assert ft_mode.has_real_rollout, (
        f"Mode {mode!r} has no rollout engines, so a fully-async soak would train off pre-recorded debug rollout "
        f"data and would prove nothing about generating while training"
    )
    assert (
        not ft_mode.colocate
    ), f"Mode {mode!r} is colocated, which train_async.py rejects: a fully-async run needs engines of its own"


if __name__ == "__main__":
    app()
