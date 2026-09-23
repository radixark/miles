# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


import asyncio
from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.cli_options import (
    FullyAsyncOption,
    MixOption,
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
from tests.utils.ft.launch import get_fully_async_args, get_train_script
from tests.utils.soak.core.config import SoakRunnerConfig, SoakTailConfig, SoakTargetConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.runner import SoakRunner
from tests.utils.soak.core.utils import (
    API_SERVER_ARGS,
    assert_fresh_dump_dir,
    compute_base_url,
    create_soak_config,
    evidence_directory,
    note_launch_outcome,
    resolve_dump_dir,
)
from tests.utils.soak.core.views import read_training_events
from tests.utils.soak.ft.actions.base import CellFaultForms
from tests.utils.soak.ft.actions.factory import compute_mean_interval_seconds_of_kind, create_cell_fault_forms
from tests.utils.soak.ft.actions.hook import HookFaultForm
from tests.utils.soak.ft.actions.remote_hook import RemoteHookFaultForm
from tests.utils.soak.ft.checkers.healing import assert_healing
from tests.utils.soak.ft.checkers.hooks import assert_hook_effects, assert_remote_p2p_failures
from tests.utils.soak.ft.checkers.survivors import assert_trainer_fault_survivors
from tests.utils.soak.ft.entrypoint import run_cell_soak
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, ROLLOUT_CELL_TYPE

from miles.utils.audit_utils.event_logger.models import Event
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector.actions.process import (
    DeadlockThreadAction,
    KillProcessAction,
    StopProcessAction,
)
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.models import FaultHookName

app: typer.Typer = typer.Typer()

TEST_NAME: str = "random_crash"

DEFAULT_SEED: int = 42
DEFAULT_NUM_STEPS: int = 60
DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS: float = 120.0
DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS: float = 240.0

HOOK_LIFETIME_SECONDS: float = 300.0
MIXED_MAX_DELAY_MS: float = 1000.0
HOOK_FAULT_ACTIONS: list[FaultAction] = [KillProcessAction(), StopProcessAction(), DeadlockThreadAction()]


@app.command(name="run")
def run_ci(
    mode: ModeOption,
    seed: SeedOption = DEFAULT_SEED,
    num_steps: NumStepsOption = DEFAULT_NUM_STEPS,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    fully_async: FullyAsyncOption = False,
    mix: MixOption = False,
) -> None:
    """Random failure soak test, for whichever components the mode enables ft on.

    Runs an async session that injects faults at random intervals via the
    api server HTTP API. The mini FT controller auto-recovers; the test passes
    if training completes without hanging.

    Doubles as the per-mode CI entry point: a CI file calls ``run_ci(mode)`` (defaults);
    manual runs use the ``run`` CLI subcommand with optional --seed/--num-steps/etc.
    """
    ft_mode: FTTestMode = resolve_mode(mode)
    if fully_async:
        assert_mode_supports_fully_async(ft_mode, mode=mode)

    config = create_soak_config(command_utils.default_config())
    test_name: str = TEST_NAME
    if mix:
        test_name += "_mixed"
    if fully_async:
        test_name += "_fully_async"
    dump_dir: str = resolve_dump_dir(f"{test_name}_{mode}", run_id=config.run_id)
    print(f"Dump directory: {dump_dir}")
    mean_interval_seconds_of_cell_type: dict[str, float] = compute_mean_interval_seconds_of_kind(
        ft_mode.ft_components,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
    )
    print(f"Seed: {seed}, Steps: {num_steps}, Mean injection intervals: {mean_interval_seconds_of_cell_type}")
    print(f"FT components: {ft_mode.ft_components}, cluster backend: {config.cluster_backend.value}")
    print(f"Train script: {get_train_script(fully_async=fully_async)}")

    prepare(ft_mode, config=config)
    train_args = _build_train_args(
        ft_mode, config=config, dump_dir=dump_dir, num_steps=num_steps, fully_async=fully_async
    )
    if mix:
        train_args += "--update-weights-timeout 600 "
    evidence_dir = evidence_directory(Path(dump_dir))
    event_log = EventLog(evidence_dir / "events.jsonl")

    injector = _run_soak(
        ft_mode,
        config=config,
        dump_dir=dump_dir,
        seed=seed,
        num_steps=num_steps,
        mean_interval_seconds_of_cell_type=mean_interval_seconds_of_cell_type,
        train_args=train_args,
        fully_async=fully_async,
        event_log=event_log,
        evidence_dir=evidence_dir,
        cell_fault_forms=_create_cell_fault_forms(ft_mode, config=config, event_log=event_log, mix=mix),
    )

    training_events = read_training_events(injector.event_log.events, dump_dir=dump_dir)
    if mix:
        _assert_hook_evidence(ft_mode, events=injector.event_log.events, training_events=training_events)
    assert_healing(
        ft_mode.ft_components,
        events=injector.event_log.events,
        forms=injector.forms,
        context=f"{test_name} {mode}",
    )

    print(f"Random failure soak test PASSED ({test_name}, mode={mode}, seed={seed}, steps={num_steps})")


def _build_train_args(
    ft_mode: FTTestMode, *, config: command_utils.ExecuteTrainConfig, dump_dir: str, num_steps: int, fully_async: bool
) -> str:
    debug_rollout_data_dir = None if ft_mode.has_real_rollout else materialize_cyclic_debug_rollout_data(num_steps)
    train_args = (
        get_common_train_args(
            ft_mode, dump_dir=dump_dir, num_steps=num_steps, debug_rollout_data_dir=debug_rollout_data_dir
        )
        + get_ft_args(ft_mode, api_server_args=API_SERVER_ARGS)
        + get_fully_async_args(fully_async=fully_async)
        + "--mini-ft-controller-enable "
    )
    if ft_mode.has_real_rollout:
        train_args += "--update-weight-transfer-mode p2p "
    assert_fresh_dump_dir(Path(dump_dir))
    return train_args


def _create_cell_fault_forms(
    ft_mode: FTTestMode, *, config: command_utils.ExecuteTrainConfig, event_log: EventLog, mix: bool
) -> CellFaultForms:
    wall_clock_forms = create_cell_fault_forms(config)
    if not mix:
        return wall_clock_forms

    base_url = compute_base_url(config)
    hook_names = [
        FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER,
        FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND,
    ]
    actor_hook_forms = [
        HookFaultForm(
            base_url=base_url,
            action=action,
            hook_name=hook_name,
            lifetime_seconds=HOOK_LIFETIME_SECONDS,
            max_delay_ms=0 if isinstance(action, DeadlockThreadAction) else MIXED_MAX_DELAY_MS,
        )
        for hook_name in hook_names
        for action in HOOK_FAULT_ACTIONS
    ]
    rollout_hook_forms = [
        RemoteHookFaultForm(
            base_url=base_url,
            hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND,
            victim=victim,
            event_log=event_log,
            lifetime_seconds=HOOK_LIFETIME_SECONDS,
            max_delay_ms=MIXED_MAX_DELAY_MS,
        )
        for victim in (wall_clock_forms[ROLLOUT_CELL_TYPE] if ft_mode.has_real_rollout else [])
    ]
    return {
        ACTOR_CELL_TYPE: [*actor_hook_forms, *wall_clock_forms[ACTOR_CELL_TYPE]],
        ROLLOUT_CELL_TYPE: [*rollout_hook_forms, *wall_clock_forms[ROLLOUT_CELL_TYPE]],
    }


def _run_soak(
    ft_mode: FTTestMode,
    *,
    config: command_utils.ExecuteTrainConfig,
    dump_dir: str,
    seed: int,
    num_steps: int,
    mean_interval_seconds_of_cell_type: dict[str, float],
    train_args: str,
    fully_async: bool,
    event_log: EventLog,
    evidence_dir: Path,
    cell_fault_forms: CellFaultForms,
) -> SoakRunner:
    expected_counts: dict[str, int] = {
        ACTOR_CELL_TYPE: ft_mode.num_cells,
        ROLLOUT_CELL_TYPE: ft_mode.rollout_num_engines,
    }
    return asyncio.run(
        run_cell_soak(
            config=config,
            dump_dir=Path(dump_dir),
            sut_run=note_launch_outcome(
                event_log=event_log,
                request_id=None,
                launching=asyncio.to_thread(
                    run_training,
                    train_args=train_args,
                    mode=ft_mode,
                    config=config,
                    train_script=get_train_script(fully_async=fully_async),
                ),
            ),
            runner_config=SoakRunnerConfig(
                seed=seed,
                target_configs={
                    kind: SoakTargetConfig(expected_count=expected_counts[kind], mean_interval_seconds=interval)
                    for kind, interval in mean_interval_seconds_of_cell_type.items()
                },
                tail=SoakTailConfig.create(num_rollout=num_steps),
            ),
            event_log=event_log,
            evidence_dir=evidence_dir,
            cell_fault_forms=cell_fault_forms,
        )
    )


def _assert_hook_evidence(ft_mode: FTTestMode, *, events: list[SoakEvent], training_events: list[Event]) -> None:
    assert_hook_effects(events, training_events=training_events)
    if "rollout" in ft_mode.ft_components:
        assert_remote_p2p_failures(events, training_events=training_events)
    if "train" in ft_mode.ft_components:
        assert_trainer_fault_survivors(events, training_events=training_events)


def assert_mode_supports_fully_async(ft_mode: FTTestMode, *, mode: str) -> None:
    assert ft_mode.has_real_rollout, (
        f"Mode {mode!r} has no rollout engines, so a fully-async soak would train off pre-recorded debug rollout "
        f"data and would prove nothing about generating while training"
    )


if __name__ == "__main__":
    app()
