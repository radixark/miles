# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import asyncio
from pathlib import Path

from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, RunSideRequest, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.comparisons import compare_deterministic_sides
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args, get_train_env_vars_arg, run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.utils.soak.core.config import SoakRunnerConfig, create_tail_policy
from tests.utils.soak.core.entrypoint import run_soak
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.scheduler import QUIESCENT_POLLS_REQUIRED
from tests.utils.soak.core.types import SoakForms
from tests.utils.soak.core.utils import (
    API_SERVER_ARGS,
    assert_fresh_dump_dir,
    compute_base_url,
    create_soak_config,
    note_launch_outcome,
)
from tests.utils.soak.ft.actions.factory import create_cell_fault_forms
from tests.utils.soak.ft.checkers.healing import assert_injections_recovered, assert_min_injections
from tests.utils.soak.ft.checkers.progress_windows import assert_faults_span_progress_windows
from tests.utils.soak.ft.observers import create_cell_observer
from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend

TEST_NAME: str = "rollout_deterministic"
NUM_ROLLOUTS: int = 8
SEED: int = 42
CRASH_INTERVAL_SECONDS: float = 30.0
POLL_INTERVAL_SECONDS: float = 0.2
HEALTH_CHECK_INTERVAL_SECONDS: float = 1.0
MIN_TRAINED_ROLLOUTS: int = 2
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
    assert not mode.colocate, f"{TEST_NAME} requires disaggregated P2P weight transfer"
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

    mean_intervals: dict[str, float] = {}
    expected_counts: dict[str, int] = {}
    forms: SoakForms = {}
    base_url = compute_base_url(config)
    if target:
        mean_intervals = {ROLLOUT_CELL_TYPE: CRASH_INTERVAL_SECONDS}
        expected_counts = {ROLLOUT_CELL_TYPE: request.mode.rollout_num_engines}
        forms = create_cell_fault_forms(base_url=base_url, config=config)

    event_log = EventLog()
    injector = asyncio.run(
        run_soak(
            config=config,
            dump_dir=Path(dump_dir),
            seed=SEED,
            mean_interval_seconds_of_kind=mean_intervals,
            expected_counts=expected_counts,
            training=note_launch_outcome(
                event_log=event_log,
                request_id=None,
                launching=asyncio.to_thread(
                    run_training, train_args=request.train_args, mode=request.mode, config=config
                ),
            ),
            runner_config=SoakRunnerConfig(
                start_after_rollout_id=0, tail=create_tail_policy(num_rollout=NUM_ROLLOUTS)
            ),
            poll_interval_seconds=POLL_INTERVAL_SECONDS,
            quiescent_polls_required=_compute_quiescent_polls_required(config),
            forms=forms,
            event_log=event_log,
            observer=create_cell_observer(
                base_url=base_url, cell_types=set(mean_intervals), forms=forms, config=config
            ),
        )
    )
    if target:
        assert_min_injections(injector.event_log.events, kind=ROLLOUT_CELL_TYPE, context=f"{TEST_NAME} rollout cells")
        assert_injections_recovered(injector.event_log.events, cell_type=ROLLOUT_CELL_TYPE, forms=forms)
        assert_faults_span_progress_windows(injector.event_log.events, dump_dir=dump_dir)


def _compute_quiescent_polls_required(config: command_utils.ExecuteTrainConfig) -> int:
    return RAY_QUIESCENT_POLLS_REQUIRED if config.cluster_backend is ClusterBackend.RAY else QUIESCENT_POLLS_REQUIRED


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
