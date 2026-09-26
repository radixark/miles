# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

from argparse import Namespace
from pathlib import Path

from tests.e2e.ft.conftest_ft.fault_hook_app import create_fault_hook_comparison_app
from tests.e2e.ft.conftest_ft.fault_hook_events import assert_weight_update_failures
from tests.e2e.ft.conftest_ft.modes import FTTestMode
from tests.utils.soak.core.utils import compute_base_url

from miles.ray.specs.inference import compute_engine_pool_id
from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.external_utils import command_utils
from miles.utils.test_utils.fault_injector.actions.process import KillProcessAction
from miles.utils.test_utils.fault_injector.actions.remote import ApiServerFaultAction
from miles.utils.test_utils.fault_injector.models import DeclaredFaultHookTarget, FaultHookName, FaultHookRequest
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.types import ClusterBackend

TEST_NAME: str = "p2p_send_receiver_fault"
NUM_ROLLOUTS: int = 8
FAULT_ROLLOUT_ID: int = 3
RECEIVER_FAULT_DELAY_MS: float = 50.0


def _build_fault_hooks(mode: FTTestMode, config: command_utils.ExecuteTrainConfig) -> list[FaultHookRequest]:
    assert config.cluster_backend is ClusterBackend.RAY, (
        f"{TEST_NAME} reaches the api server from the sending trainer rank, and only the ray backend serves it on "
        f"the trainer's own host, not the {config.cluster_backend.value} backend"
    )
    sender_cell_id: str = compute_cell_id(pool_id=compute_trainer_pool_id("actor"), cell_index=0)
    return [
        FaultHookRequest(
            request_id=f"kill_receiver_before_send_at_{FAULT_ROLLOUT_ID}",
            hook_name=FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND,
            action=ApiServerFaultAction(
                base_url=compute_base_url(config),
                cell_id=_receiver_cell_id(config),
                rank=0,
                inner=KillProcessAction(),
            ),
            target=DeclaredFaultHookTarget(cell_id=sender_cell_id, rank=0),
            rollout_id=FAULT_ROLLOUT_ID,
            delay_ms=RECEIVER_FAULT_DELAY_MS,
        )
    ]


def _receiver_cell_id(config: command_utils.ExecuteTrainConfig) -> str:
    deploy_args = Namespace(
        deploy_instance_id=config.deploy_instance_id, deploy_component=config.deploy_component.value
    )
    return compute_cell_id(pool_id=compute_engine_pool_id(deploy_args, model_idx=0, group_index=0), cell_index=0)


def _assert_target_events(events_dir: Path, mode: FTTestMode) -> None:
    assert_weight_update_failures(
        events_dir,
        failed_cell_ids_of_rollout_id={FAULT_ROLLOUT_ID: [_receiver_cell_id(command_utils.default_config())]},
    )


app, run_ci = create_fault_hook_comparison_app(
    test_name=TEST_NAME,
    num_rollouts=NUM_ROLLOUTS,
    ft_components=("rollout",),
    extra_train_args="",
    build_fault_hooks=_build_fault_hooks,
    expected_target_reconfigures=lambda mode: [],
    assert_target_events=_assert_target_events,
)

if __name__ == "__main__":
    app()
