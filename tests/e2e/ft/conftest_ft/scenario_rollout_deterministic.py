# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import asyncio
import contextlib
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, RunSideRequest, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.comparisons import compare_deterministic_sides
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args, get_train_env_vars_arg, run_training
from tests.e2e.ft.conftest_ft.fault_injection.entrypoint import (
    API_SERVER_PORT,
    FaultInjectorHandle,
    spawn_fault_injector,
)
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import create_cell_fault_forms
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_injection_times, compute_num_injections
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.e2e.ft.conftest_ft.scenario_random_crash import assert_rollout_cells_served_after_injection
from tests.utils.soak.core.config import (
    QUIESCENT_POLLS_REQUIRED,
    SoakRunnerConfig,
    SoakTailConfig,
    SoakTargetConfig,
)
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.utils import (
    API_SERVER_ARGS,
    assert_fresh_dump_dir,
    create_soak_config,
    evidence_directory,
    note_launch_outcome,
)
from tests.utils.soak.ft.checkers.healing import assert_injections_recovered, assert_min_injections
from tests.utils.soak.ft.checkers.progress_windows import (
    MIN_FAULT_PROGRESS_WINDOWS,
    _compute_fault_progress_windows,
    assert_faults_span_progress_windows,
)
from tests.utils.soak.ft.checkers.reconfigure import assert_min_soak_injections
from tests.utils.soak.ft.entrypoint import run_cell_soak
from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE

from miles.utils.external_utils import command_utils
from miles.utils.misc import MutableBox
from miles.utils.test_utils.comparisons.metrics import read_rollout_completion_times
from miles.utils.workers.types import ClusterBackend

TEST_NAME: str = "rollout_deterministic"
NUM_ROLLOUTS: int = 8
SEED: int = 42
CRASH_INTERVAL_SECONDS: float = 30.0
POLL_INTERVAL_SECONDS: float = 0.2
HEALTH_CHECK_INTERVAL_SECONDS: float = 1.0
MIN_TRAINED_ROLLOUTS: int = 2
FIRST_ROLLOUT_TIMEOUT_SECONDS: float = 3600.0
FIRST_ROLLOUT_POLL_SECONDS: float = 5.0
TERMINAL_FAULT_FREE_ROLLOUTS: int = 2
RAY_QUIESCENT_POLLS_REQUIRED: int = 1


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
    assert tuple(mode.ft_components) == ("rollout",), (
        f"{TEST_NAME} injects into rollout cells only, so the mode must enable ft on rollout alone, "
        f"got ft_components={mode.ft_components}"
    )

    args = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_ROLLOUTS, enable_dumper=enable_dumper)
    args += get_ft_args(mode, api_server_args=API_SERVER_ARGS)
    args += "--mini-ft-controller-enable "
    args += "--debug-deterministic-collective "
    args += "--sglang-disable-radix-cache "
    args += "--update-weight-transfer-mode p2p --sglang-router-policy round_robin "
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
    assert_fresh_dump_dir(Path(dump_dir))

    target_configs: dict[str, SoakTargetConfig] = {}
    if target:
        target_configs = {
            ROLLOUT_CELL_TYPE: SoakTargetConfig(
                expected_count=request.mode.rollout_num_engines, mean_interval_seconds=CRASH_INTERVAL_SECONDS
            )
        }

    evidence_dir = evidence_directory(Path(dump_dir))
    event_log = EventLog(evidence_dir / "events.jsonl")
    injector = asyncio.run(
        run_cell_soak(
            config=config,
            dump_dir=Path(dump_dir),
            sut_run=note_launch_outcome(
                event_log=event_log,
                request_id=None,
                launching=asyncio.to_thread(
                    run_training, train_args=request.train_args, mode=request.mode, config=config
                ),
            ),
            runner_config=SoakRunnerConfig(
                seed=SEED,
                start_after_rollout_id=0,
                target_configs=target_configs,
                tail=SoakTailConfig.create(num_rollout=NUM_ROLLOUTS),
                poll_interval_seconds=POLL_INTERVAL_SECONDS,
                quiescent_polls_required=_compute_quiescent_polls_required(config),
            ),
            event_log=event_log,
            evidence_dir=evidence_dir,
        )
    )
    if target:
        assert_min_injections(injector.event_log.events, kind=ROLLOUT_CELL_TYPE, context=f"{TEST_NAME} rollout cells")
        assert_injections_recovered(injector.event_log.events, cell_type=ROLLOUT_CELL_TYPE, forms=injector.forms)
        assert_faults_span_progress_windows(injector.event_log.events, dump_dir=dump_dir)


@contextlib.contextmanager
def _inject_rollout_faults(
    mode: FTTestMode, dump_dir: str, config: command_utils.ExecuteTrainConfig
) -> Iterator[None]:
    base_url: str = f"http://{config.create_backend().api_server_host(config)}:{API_SERVER_PORT}"
    print(f"Injecting into {ROLLOUT_CELL_TYPE} cells only, mean interval {CRASH_INTERVAL_SECONDS:.1f}s, seed {SEED}")

    shutil.rmtree(dump_dir, ignore_errors=True)

    armed: MutableBox[FaultInjectorHandle | None] = MutableBox(value=None)

    def arm_on_generation_start() -> None:
        if not _wait_for_first_rollout(dump_dir):
            return
        armed.value = spawn_fault_injector(
            base_url=base_url,
            seed=SEED,
            mean_interval_seconds_of_cell_type={ROLLOUT_CELL_TYPE: CRASH_INTERVAL_SECONDS},
            cell_fault_forms=create_cell_fault_forms(base_url=base_url, config=config),
            injection_enabled=lambda: _rollout_fault_injection_enabled(dump_dir),
            poll_interval_seconds=POLL_INTERVAL_SECONDS,
            quiescent_polls_required=_compute_quiescent_polls_required(config),
        )

    arming = threading.Thread(target=arm_on_generation_start, daemon=True, name="ft-rollout-injector-arm")
    arming.start()
    try:
        yield
    finally:
        arming.join(timeout=FIRST_ROLLOUT_POLL_SECONDS)
        if armed.value is not None:
            armed.value.stop_and_join()

    injector = armed.value
    assert injector is not None, (
        f"No injector was ever armed: the target never reported a finished rollout within "
        f"{FIRST_ROLLOUT_TIMEOUT_SECONDS:.0f}s, so nothing was crashed and the comparison would be vacuous"
    )
    assert_min_soak_injections(
        compute_num_injections(injector.event_log.events, cell_type=ROLLOUT_CELL_TYPE),
        context=f"{TEST_NAME} rollout cells",
    )
    assert_rollout_cells_served_after_injection(injector)
    _assert_injections_spread_over_rollouts(injector, dump_dir=dump_dir)


def _compute_quiescent_polls_required(config: command_utils.ExecuteTrainConfig) -> int:
    return RAY_QUIESCENT_POLLS_REQUIRED if config.cluster_backend is ClusterBackend.RAY else QUIESCENT_POLLS_REQUIRED


def _wait_for_first_rollout(dump_dir: str) -> bool:
    deadline = time.monotonic() + FIRST_ROLLOUT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if read_rollout_completion_times(dump_dir):
            return True
        time.sleep(FIRST_ROLLOUT_POLL_SECONDS)
    return False


def _rollout_fault_injection_enabled(dump_dir: str) -> bool:
    completed_rollout_ids: set[int] = {rollout_id for rollout_id, _ in read_rollout_completion_times(dump_dir)}
    next_rollout_id = max(completed_rollout_ids, default=-1) + 1
    return next_rollout_id < NUM_ROLLOUTS - TERMINAL_FAULT_FREE_ROLLOUTS


def _assert_injections_spread_over_rollouts(injector: FaultInjectorHandle, *, dump_dir: str) -> None:
    crashed_rollouts = _compute_fault_progress_windows(
        injected_at=compute_injection_times(injector.event_log.events, cell_type=ROLLOUT_CELL_TYPE),
        rollout_completions=read_rollout_completion_times(dump_dir),
    )

    assert len(crashed_rollouts) >= MIN_FAULT_PROGRESS_WINDOWS, (
        f"Every accepted injection landed inside rollout(s) {sorted(crashed_rollouts)}, so this run only shows "
        f"that {len(crashed_rollouts)} rollout survived a crash rather than that crashes cost the loss curve "
        f"nothing across the run"
    )
    print(f"Injections landed across rollouts {sorted(crashed_rollouts)}")


def _compare(dump_dir: str, mode: FTTestMode) -> None:
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
