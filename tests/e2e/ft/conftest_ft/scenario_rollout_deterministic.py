# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import json
import shlex
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, RunSideRequest, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.comparisons import compare_deterministic_sides
from tests.e2e.ft.conftest_ft.execution import (
    _DETERMINISTIC_ENV_VARS,
    get_common_train_args,
    get_ft_args,
    get_train_env_vars_arg,
    run_training,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.utils.soak.checks.determinism import assert_deterministic_environment
from tests.utils.soak.checks.ft import assert_rollout_cells_served_after_injection
from tests.utils.soak.checks.tail import assert_tail_complete
from tests.utils.soak.checks.weights import assert_published_weight_checksums
from tests.utils.soak.config import SoakCellPolicy, SoakPolicy, create_tail_policy
from tests.utils.soak.entrypoint import API_SERVER_PORT, FaultInjectorHandle, spawn_fault_injector
from tests.utils.soak.fault_forms import ROLLOUT_CELL_TYPE, create_cell_fault_forms
from tests.utils.soak.state import event_source
from tests.utils.soak.storage import validate_dump_storage
from tests.utils.soak.teardown import teardown_run
from tests.utils.soak.utils import create_soak_config, evidence_directory, get_api_server_args
from tests.utils.soak.views import compute_injection_times, compute_num_injections

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.comparisons.inference_engine_checksums import assert_identified_engine_checksums
from miles.utils.test_utils.comparisons.metrics import read_rollout_completion_times
from miles.utils.test_utils.reconfigure_assertions import assert_min_soak_injections

TEST_NAME: str = "rollout_deterministic"
NUM_ROLLOUTS: int = 8
SEED: int = 42
CRASH_INTERVAL_SECONDS: float = 30.0
POLL_INTERVAL_SECONDS: float = 0.2
HEALTH_CHECK_INTERVAL_SECONDS: float = 1.0
MIN_TRAINED_ROLLOUTS: int = 2
MIN_FAULT_PROGRESS_WINDOWS: int = 2
TERMINAL_FAULT_FREE_ROLLOUTS: int = 3


DETERMINISTIC_INFERENCE_ENV_VARS: dict[str, str] = {
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "false",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "false",
}


def _build_args(
    mode: FTTestMode,
    dump_dir: str,
    enable_dumper: bool = True,
    config: command_utils.ExecuteTrainConfig | None = None,
) -> str:
    assert mode.has_real_rollout, f"{TEST_NAME} needs engines to crash, but mode {mode.model_name} has none"
    assert not mode.colocate, f"{TEST_NAME} requires disaggregated P2P weight transfer"
    assert tuple(mode.ft_components) == ("rollout",), (
        f"{TEST_NAME} injects into rollout cells only, so the mode must enable ft on rollout alone, "
        f"got ft_components={mode.ft_components}"
    )

    args = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_ROLLOUTS, enable_dumper=enable_dumper)
    args += get_ft_args(mode, api_server_args=get_api_server_args(config))
    args += "--mini-ft-controller-enable "
    args += "--debug-deterministic-collective "
    args += "--sglang-disable-radix-cache "
    args += "--update-weight-transfer-mode p2p --sglang-router-policy round_robin "
    args += f"--inference-env-vars {shlex.quote(json.dumps(DETERMINISTIC_INFERENCE_ENV_VARS))} "
    args += f"--rollout-health-check-interval {HEALTH_CHECK_INTERVAL_SECONDS} "
    args += "--weight-decay 0 "
    args += get_train_env_vars_arg(
        mode,
        deterministic=True,
        extra_env_vars=DETERMINISTIC_INFERENCE_ENV_VARS,
    )
    return args


def _run_side(request: RunSideRequest) -> None:
    config = request.config
    dump_dir = request.dump_dir
    target = request.side == TARGET_SIDE
    validate_dump_storage(Path(dump_dir))
    if Path(dump_dir).exists():
        shutil.rmtree(dump_dir)
    base_url: str = f"http://{config.create_backend().api_server_host(config)}:{API_SERVER_PORT}"
    evidence_dir = evidence_directory(Path(dump_dir))
    injector = spawn_fault_injector(
        evidence_path=evidence_dir / "events.jsonl",
        sources={"training_events": Path(dump_dir) / EVENTS_DIRNAME},
        config=config,
        base_url=base_url,
        seed=SEED,
        mean_interval_seconds_of_cell_type={ROLLOUT_CELL_TYPE: CRASH_INTERVAL_SECONDS} if target else {},
        cell_fault_forms=create_cell_fault_forms(base_url=base_url, config=config) if target else {},
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        policy=SoakPolicy(
            start_after_rollout_id=0,
            cell_policies=(
                {ROLLOUT_CELL_TYPE: SoakCellPolicy(expected_cells=request.mode.rollout_num_engines)} if target else {}
            ),
        ),
        tail_policy=create_tail_policy(num_rollout=NUM_ROLLOUTS, min_tail_rollouts=TERMINAL_FAULT_FREE_ROLLOUTS),
    )
    try:
        run_training(
            train_args=request.train_args,
            mode=request.mode,
            dump_dir=dump_dir,
            config=config,
            injector=injector,
        )
    finally:
        injector.stop_and_join(
            teardown=partial(teardown_run, config=config, event_log=injector.event_log, evidence_dir=evidence_dir)
        )
    assert_tail_complete(injector.event_log.events)
    if target:
        assert_min_soak_injections(
            compute_num_injections(injector.event_log.events, cell_type=ROLLOUT_CELL_TYPE),
            context=f"{TEST_NAME} rollout cells",
        )
        assert_rollout_cells_served_after_injection(injector)
        _assert_faults_span_progress_windows(injector, dump_dir=dump_dir)


def _assert_faults_span_progress_windows(injector: FaultInjectorHandle, *, dump_dir: str) -> None:
    source = event_source(injector.event_log.events, name="training_events", fallback=Path(dump_dir) / EVENTS_DIRNAME)
    progress_windows = _compute_fault_progress_windows(
        injected_at=compute_injection_times(injector.event_log.events, cell_type=ROLLOUT_CELL_TYPE),
        rollout_completions=read_rollout_completion_times(str(source.parent)),
    )

    assert len(progress_windows) >= MIN_FAULT_PROGRESS_WINDOWS, (
        f"Fault effects occupy only {sorted(progress_windows)} progress windows; "
        f"expected at least {MIN_FAULT_PROGRESS_WINDOWS} windows separated by completed rollouts"
    )
    print(f"Fault effects span progress windows {sorted(progress_windows)}")


def _compute_fault_progress_windows(
    *, injected_at: list[datetime], rollout_completions: list[tuple[int, datetime]]
) -> set[int]:
    return {
        max((rollout_id for rollout_id, finished_at in rollout_completions if finished_at <= at), default=-1) + 1
        for at in injected_at
    }


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    for side in (BASELINE_SIDE, TARGET_SIDE):
        assert_identified_engine_checksums(dump_dir=Path(dump_dir) / side)
        assert_published_weight_checksums(read_events(Path(dump_dir) / side / EVENTS_DIRNAME))
        assert_deterministic_environment(
            read_events(Path(dump_dir) / side / EVENTS_DIRNAME),
            trainer_ranks={(0, rank) for rank in range(mode.train_num_nodes * mode.train_gpus_per_node)},
            engine_count=mode.rollout_num_engines,
            engine_env=DETERMINISTIC_INFERENCE_ENV_VARS,
            trainer_env=_DETERMINISTIC_ENV_VARS,
        )
    compare_deterministic_sides(
        baseline_dir=f"{dump_dir}/{BASELINE_SIDE}",
        target_dir=f"{dump_dir}/{TARGET_SIDE}",
        min_trained_rollouts=MIN_TRAINED_ROLLOUTS,
    )

    print("Rollout ft deterministic comparison test PASSED")


app, run_ci = create_comparison_app_and_run_ci(
    test_name=TEST_NAME,
    build_baseline_args=_build_args,
    build_target_args=_build_args,
    compare_fn=_compare,
    config_for_side=lambda side, config: create_soak_config(config),
    run_side=_run_side,
    release_side=lambda request: None,
)

if __name__ == "__main__":
    app()
