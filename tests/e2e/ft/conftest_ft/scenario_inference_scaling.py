# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.e2e.ft.conftest_ft.scaling import (
    SCALING_NUM_ROLLOUTS,
    compute_scaling_mode,
    compute_scaling_schedule,
    run_scaling_scenario,
)
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.ft.actions.resize import ScalingStep
from tests.utils.soak.ft.checkers.reconfigure import assert_reconfigure_events
from tests.utils.soak.ft.checkers.resize import assert_counts_follow_schedule, read_engine_counts_of_rollout
from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE

from miles.ray.specs.inference import ENGINE_POOL_ID_PREFIX
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.test_utils.comparisons.metrics import read_metric_events
from miles.utils.workers.types import DeployComponent

app: typer.Typer = typer.Typer()

TEST_NAME: str = "inference_scaling"
SCHEDULE: tuple[ScalingStep, ...] = compute_scaling_schedule("training")

_MODE: FTTestMode = compute_scaling_mode(("rollout",))
_POOL_ID: str = f"{ENGINE_POOL_ID_PREFIX}-{DeployComponent.ALL.value}-0-0"

_NUM_SAMPLES_KEY: str = "rollout/num_training_samples"
_MEAN_RESPONSE_LENGTH_KEY: str = "rollout/response_len/mean"
_ROLLOUT_TIME_KEY: str = "perf/rollout_time"
_TOKENS_PER_GPU_PER_SEC_KEY: str = "perf/effective_tokens_per_gpu_per_sec"


@app.command(name="run")
def run_ci() -> None:
    run_scaling_scenario(
        test_name=TEST_NAME,
        mode=_MODE,
        schedule=SCHEDULE,
        pool_id=_POOL_ID,
        cell_type=ROLLOUT_CELL_TYPE,
        assert_scaled=_assert_engines_scaled,
    )


def _assert_engines_scaled(events: list[SoakEvent], dump_dir: str) -> None:
    initial = _MODE.rollout_num_engines

    engine_counts = read_engine_counts_of_rollout(dump_dir)
    assert engine_counts.get(-1) == initial, (
        f"the weight update before rollout 0 reached {engine_counts.get(-1)} engine(s), not the {initial} the "
        f"run was deployed with"
    )
    assert_counts_follow_schedule(
        {rollout_id: count for rollout_id, count in engine_counts.items() if rollout_id >= 0},
        initial=initial,
        schedule=SCHEDULE,
        num_rollouts=SCALING_NUM_ROLLOUTS,
        what="the number of engines a weight update reached",
    )
    assert_counts_follow_schedule(
        _read_gpu_counts_of_rollout(dump_dir),
        initial=initial * _MODE.rollout_gpus_per_engine,
        schedule=SCHEDULE,
        num_rollouts=SCALING_NUM_ROLLOUTS,
        what="the engine gpu count the rollout throughput was normalized by",
    )
    assert_reconfigure_events(Path(dump_dir) / EVENTS_DIRNAME, expected=[])


def _read_gpu_counts_of_rollout(dump_dir: str) -> dict[int, int]:
    keys = (_NUM_SAMPLES_KEY, _MEAN_RESPONSE_LENGTH_KEY, _ROLLOUT_TIME_KEY, _TOKENS_PER_GPU_PER_SEC_KEY)
    counts: dict[int, int] = {}
    for event in read_metric_events(Path(dump_dir) / EVENTS_DIRNAME):
        if event.rollout_id is None or not all(key in event.metrics for key in keys):
            continue
        assert (
            event.rollout_id not in counts
        ), f"{dump_dir} logs the rollout metrics of rollout {event.rollout_id} twice"
        samples, response_length, rollout_time, tokens_per_gpu_per_sec = (event.metrics[key] for key in keys)
        counts[event.rollout_id] = round(samples * response_length / rollout_time / tokens_per_gpu_per_sec)
    return counts


if __name__ == "__main__":
    app()
