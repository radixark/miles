# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import contextlib
import dataclasses
import math
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from tests.e2e.ft.conftest_ft.app import create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args, get_train_env_vars_arg
from tests.e2e.ft.conftest_ft.fault_injection.entrypoint import (
    API_SERVER_PORT,
    FaultInjectorHandle,
    spawn_fault_injector,
)
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ROLLOUT_CELL_TYPE, create_cell_fault_forms
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_injection_times, compute_num_injections
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.e2e.ft.conftest_ft.scenario_random_crash import assert_every_rollout_injection_recovered

from miles.utils.external_utils import command_utils
from miles.utils.test_utils.comparisons.dumps import (
    INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    INPUT_TENSORS_SKIP_PATTERN,
    compare_dumps,
)
from miles.utils.test_utils.comparisons.inference_engine_checksums import (
    assert_engine_weights_moved,
    compare_inference_engine_checksums,
)
from miles.utils.test_utils.comparisons.metrics import (
    assert_metrics_classified,
    compare_metrics,
    read_metric_series,
    read_rollout_completion_times,
)
from miles.utils.test_utils.reconfigure_assertions import assert_min_soak_injections, assert_reconfigure_events

TEST_NAME: str = "rollout_deterministic"
NUM_ROLLOUTS: int = 8
COMPARED_METRIC_PREFIXES: tuple[str, ...] = ("train/", "rollout/")
UNCOMPARED_METRIC_PREFIXES: tuple[str, ...] = ("perf/",)
SEED: int = 42
CRASH_INTERVAL_SECONDS: float = 120.0
HEALTH_CHECK_INTERVAL_SECONDS: float = 5.0
DETERMINISTIC_INFERENCE_ENV_VARS: dict[str, str] = {"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "false"}
MIN_TRAINED_ROLLOUTS: int = 2
FIRST_ROLLOUT_TIMEOUT_SECONDS: float = 3600.0
FIRST_ROLLOUT_POLL_SECONDS: float = 5.0
MIN_CRASHED_ROLLOUTS: int = 2


COLOCATED_MEM_FRACTION_STATIC: float = 0.4
DETERMINISTIC_INFERENCE_ENV_VARS: dict[str, str] = {"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "false"}


def _build_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    assert mode.has_real_rollout, f"{TEST_NAME} needs engines to crash, but mode {mode.model_name} has none"
    assert tuple(mode.ft_components) == ("rollout",), (
        f"{TEST_NAME} injects into rollout cells only, so the mode must enable ft on rollout alone, "
        f"got ft_components={mode.ft_components}"
    )

    args = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_ROLLOUTS, enable_dumper=enable_dumper)
    args += get_ft_args(mode)
    args += f"--api-server-port {API_SERVER_PORT} --mini-ft-controller-enable "
    args += "--debug-deterministic-collective "
    args += "--sglang-disable-radix-cache "
    if mode.colocate:
        args += f"--sglang-mem-fraction-static {COLOCATED_MEM_FRACTION_STATIC} "
    args += f"--rollout-health-check-interval {HEALTH_CHECK_INTERVAL_SECONDS} "
    args += "--weight-decay 0 "
    args += get_train_env_vars_arg(
        mode,
        deterministic=True,
        extra_env_vars=DETERMINISTIC_INFERENCE_ENV_VARS,
    )
    return args


@contextlib.contextmanager
def _inject_rollout_faults(
    mode: FTTestMode, dump_dir: str, config: command_utils.ExecuteTrainConfig
) -> Iterator[None]:
    base_url: str = f"http://{config.create_backend().api_server_host()}:{API_SERVER_PORT}"
    print(f"Injecting into {ROLLOUT_CELL_TYPE} cells only, mean interval {CRASH_INTERVAL_SECONDS:.1f}s, seed {SEED}")

    armed = _MutableBox()

    def arm_on_generation_start() -> None:
        if not _wait_for_first_rollout(dump_dir):
            return
        armed.value = spawn_fault_injector(
            base_url=base_url,
            seed=SEED,
            mean_interval_seconds_of_cell_type={ROLLOUT_CELL_TYPE: CRASH_INTERVAL_SECONDS},
            cell_fault_forms=create_cell_fault_forms(base_url=base_url, config=config),
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
    assert_every_rollout_injection_recovered(injector)
    _assert_injections_spread_over_rollouts(injector, dump_dir=dump_dir)


@dataclasses.dataclass
class _MutableBox:
    value: FaultInjectorHandle | None = None


def _wait_for_first_rollout(dump_dir: str) -> bool:
    deadline = time.monotonic() + FIRST_ROLLOUT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if read_rollout_completion_times(dump_dir):
            return True
        time.sleep(FIRST_ROLLOUT_POLL_SECONDS)
    return False


def _assert_injections_spread_over_rollouts(injector: FaultInjectorHandle, *, dump_dir: str) -> None:
    crashed_rollouts = _compute_crashed_rollouts(
        injected_at=compute_injection_times(injector.event_log.events, cell_type=ROLLOUT_CELL_TYPE),
        rollout_completions=read_rollout_completion_times(dump_dir),
    )

    assert len(crashed_rollouts) >= MIN_CRASHED_ROLLOUTS, (
        f"Every accepted injection landed inside rollout(s) {sorted(crashed_rollouts)}, so this run only shows "
        f"that {len(crashed_rollouts)} rollout survived a crash rather than that crashes cost the loss curve "
        f"nothing across the run"
    )
    print(f"Injections landed across rollouts {sorted(crashed_rollouts)}")


def _compute_crashed_rollouts(
    *, injected_at: list[datetime], rollout_completions: list[tuple[int, datetime]]
) -> set[int]:
    return {sum(1 for _, finished_at in rollout_completions if finished_at <= at) for at in injected_at}


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    baseline_dir: str = f"{dump_dir}/baseline"
    target_dir: str = f"{dump_dir}/target"

    for side_dir in (baseline_dir, target_dir):
        assert_reconfigure_events(Path(f"{side_dir}/events"), expected=[])

    for side_dir in (baseline_dir, target_dir):
        assert_metrics_classified(side_dir, compared=COMPARED_METRIC_PREFIXES, ignored=UNCOMPARED_METRIC_PREFIXES)
    compare_metrics(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        rtol=0.0,
        atol=0.0,
        key_prefixes=list(COMPARED_METRIC_PREFIXES),
        exclude_keys=[],
    )

    compare_dumps(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        diff_thresholds=[(".*", "rel <= 0")],
        allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
        allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    )

    compare_inference_engine_checksums(baseline_dir=baseline_dir, target_dir=target_dir)

    for side, side_dir in (("baseline", baseline_dir), ("target", target_dir)):
        assert_engine_weights_moved(side=side, dump_dir=side_dir)
        _assert_gradients_were_nonzero(side=side, dump_dir=side_dir)

    print("Rollout ft deterministic comparison test PASSED")


def _assert_gradients_were_nonzero(*, side: str, dump_dir: str) -> None:
    series = read_metric_series(dump_dir, key="train/grad_norm")
    usable = [(rollout_id, value) for rollout_id, value in series if math.isfinite(value) and value != 0.0]

    assert len(usable) >= MIN_TRAINED_ROLLOUTS, (
        f"{side}: train/grad_norm is finite and non-zero in only {len(usable)} of {len(series)} rollout(s) "
        f"({series}), so this run's weights could have moved on weight decay alone"
    )


app, run_ci = create_comparison_app_and_run_ci(
    test_name=TEST_NAME,
    build_baseline_args=_build_args,
    build_target_args=_build_args,
    compare_fn=_compare,
    target_side_context=_inject_rollout_faults,
)

if __name__ == "__main__":
    app()
