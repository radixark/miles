# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

from pathlib import Path

import typer
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.e2e.ft.conftest_ft.scaling import SCALING_SCHEDULE, compute_scaling_mode, run_scaling_scenario
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.views import read_training_events
from tests.utils.soak.ft.checkers.resize import LANDING_LAG_ROLLOUTS
from tests.utils.soak.ft.types import ACTOR_CELL_TYPE, compute_sizes

from miles.backends.megatron_utils.ft.types import TrainStepOutcome
from miles.backends.megatron_utils.megatron_config import ACTOR_ROLE
from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.audit_utils.event_logger.models import TrainGroupStepEndEvent

app: typer.Typer = typer.Typer()

_TEST_NAME: str = "trainer_scaling"
_MODE: FTTestMode = compute_scaling_mode(("train",))


@app.command(name="run")
def run_ci() -> None:
    run_scaling_scenario(
        test_name=_TEST_NAME,
        mode=_MODE,
        pool_id=compute_trainer_pool_id(ACTOR_ROLE),
        cell_type=ACTOR_CELL_TYPE,
        scaling_flag="--actor-num-gpus-per-node",
        num_gpus_per_cell=_MODE.train_gpus_per_node // _MODE.num_cells,
        read_counts_of_rollout_id=_read_num_trained_cells_of_rollout_id,
        counted="the number of trainer cells that trained a step",
    )


def _read_num_trained_cells_of_rollout_id(events: list[SoakEvent], dump_dir: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for event in read_training_events(events, dump_dir=dump_dir):
        if not isinstance(event, TrainGroupStepEndEvent):
            continue
        assert event.rollout_id not in counts, f"{dump_dir} ends rollout {event.rollout_id} twice"

        removed = _compute_removed_cell_indices(event.rollout_id)
        assert all(
            cell_index in removed if outcomes == "error" else all(one is TrainStepOutcome.NORMAL for one in outcomes)
            for cell_index, outcomes in event.cell_outcomes.items()
        ), (
            f"attempt {event.attempt} of rollout {event.rollout_id} ended with {event.cell_outcomes}, and only the "
            f"cell(s) {sorted(removed)} a shrink removes around then may fail in it, so a cell that a resize added "
            f"or removed took the step down with it instead of joining or leaving cleanly"
        )
        counts[event.rollout_id] = len(event.cell_outcomes)
    return counts


def _compute_removed_cell_indices(rollout_id: int) -> set[int]:
    sizes = compute_sizes(initial_replicas=_MODE.num_cells, schedule=SCALING_SCHEDULE)
    return {
        cell_index
        for step, size_before in zip(SCALING_SCHEDULE, sizes[:-1], strict=True)
        if step.at_rollout <= rollout_id <= step.at_rollout + LANDING_LAG_ROLLOUTS
        for cell_index in range(step.replicas, size_before)
    }


if __name__ == "__main__":
    app()
