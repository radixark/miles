# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.e2e.ft.conftest_ft.scaling import SCALING_NUM_ROLLOUTS, compute_scaling_mode, run_scaling_scenario
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import read_training_events
from tests.utils.soak.ft.types import ROLLOUT_CELL_TYPE

from miles.ray.specs.inference import ENGINE_POOL_ID_PREFIX
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.workers.types import DeployComponent

app: typer.Typer = typer.Typer()

_TEST_NAME: str = "inference_scaling"
_MODE: FTTestMode = compute_scaling_mode(("rollout",))
_POOL_ID: str = f"{ENGINE_POOL_ID_PREFIX}-{DeployComponent.ALL.value}-0-0"


@app.command(name="run")
def run_ci() -> None:
    run_scaling_scenario(
        test_name=_TEST_NAME,
        mode=_MODE,
        pool_id=_POOL_ID,
        cell_type=ROLLOUT_CELL_TYPE,
        scaling_flag="--rollout-num-gpus",
        num_gpus_per_cell=_MODE.rollout_gpus_per_engine,
        read_counts_of_rollout_id=_read_engine_counts_of_rollout_id,
        counted="the number of engines the weights a rollout generated with reached",
    )


def _read_engine_counts_of_rollout_id(events: list[SoakEvent], dump_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for event in read_training_events(events, dump_dir=dump_dir):
        if isinstance(event, InferenceEngineWeightChecksumEvent):
            assert (
                event.rollout_id + 1 not in counts
            ), f"{dump_dir} holds two weight updates for rollout {event.rollout_id}"
            counts[event.rollout_id + 1] = len(event.engine_snapshots)
    return {rollout_id: count for rollout_id, count in counts.items() if rollout_id < SCALING_NUM_ROLLOUTS}


if __name__ == "__main__":
    app()
