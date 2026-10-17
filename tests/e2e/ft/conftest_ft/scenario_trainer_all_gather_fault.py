# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

from pathlib import Path

from tests.e2e.ft.conftest_ft.fault_hook_app import create_fault_hook_comparison_app
from tests.e2e.ft.conftest_ft.fault_hook_events import assert_weight_updates_published
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.utils.soak.ft.checkers.reconfigure import ReconfigureInfo

from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.test_utils.fault_injector.actions.process import (
    DeadlockThreadAction,
    KillProcessAction,
    StopProcessAction,
)
from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.models import DeclaredFaultHookTarget, FaultHookName, FaultHookRequest
from miles.utils.workers.naming import compute_cell_id

TEST_NAME: str = "trainer_all_gather_fault"
NUM_ROLLOUTS: int = 8
UPDATE_WEIGHTS_TIMEOUT_SECONDS: float = 120.0
FAULT_ACTION_OF_ROLLOUT_ID: dict[int, FaultAction] = {
    1: KillProcessAction(),
    3: StopProcessAction(),
    5: DeadlockThreadAction(),
}


def _build_fault_hooks(mode: FTTestMode, config: ExecuteTrainConfig) -> list[FaultHookRequest]:
    target_cell_id: str = compute_cell_id(pool_id=compute_trainer_pool_id("actor"), cell_index=mode.num_cells - 1)
    return [
        FaultHookRequest(
            request_id=f"{action.kind}_before_all_gather_at_{rollout_id}",
            hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_ALL_GATHER,
            action=action,
            target=DeclaredFaultHookTarget(cell_id=target_cell_id, rank=0),
            rollout_id=rollout_id,
        )
        for rollout_id, action in FAULT_ACTION_OF_ROLLOUT_ID.items()
    ]


def _expected_reconfigures(mode: FTTestMode) -> list[ReconfigureInfo]:
    return [
        ReconfigureInfo(
            rollout_id=rollout_id + 1,
            src_cell_index=0,
            healed_cell_indices=[mode.num_cells - 1],
            alive_cell_indices_after=list(range(mode.num_cells)),
        )
        for rollout_id in FAULT_ACTION_OF_ROLLOUT_ID
    ]


def _assert_target_events(events_dir: Path, mode: FTTestMode) -> None:
    assert_weight_updates_published(events_dir, rollout_ids=FAULT_ACTION_OF_ROLLOUT_ID)


app, run_ci = create_fault_hook_comparison_app(
    test_name=TEST_NAME,
    num_rollouts=NUM_ROLLOUTS,
    ft_components=("train",),
    extra_train_args=f"--update-weights-timeout {UPDATE_WEIGHTS_TIMEOUT_SECONDS} ",
    build_fault_hooks=_build_fault_hooks,
    expected_target_reconfigures=_expected_reconfigures,
    assert_target_events=_assert_target_events,
)

if __name__ == "__main__":
    app()
