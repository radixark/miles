import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.ray.train.test_group import _make_controller

from miles.ray.train.init_request import TrainerControllerInitRequest


class TestControllerRuntimeInputs:
    @pytest.mark.parametrize(
        "num_rollout,wandb_run_id,mlflow_run_id", [(19, "wandb-new", "mlflow-new"), (None, None, None)]
    )
    async def test_initialization_updates_runtime_fields_on_the_original_trainer_payload(
        self, num_rollout: int | None, wandb_run_id: str | None, mlflow_run_id: str | None
    ) -> None:
        """Runtime IDs update the deployed trainer payload without replacing its configuration."""
        controller = _make_controller(num_cells=1)
        args = controller.args
        args.num_rollout = 3
        args.wandb_run_id = "wandb-old"
        args.mlflow_run_id = "mlflow-old"
        request = TrainerControllerInitRequest(
            num_rollout=num_rollout, wandb_run_id=wandb_run_id, mlflow_run_id=mlflow_run_id
        )

        try:
            await controller.init(request)

            assert controller.args is args
            assert args.num_rollout == num_rollout
            assert args.wandb_run_id == wandb_run_id
            assert args.mlflow_run_id == mlflow_run_id
            assert args.trainer_id == "actor"
            assert args.trainer_init_expected_num_cells == 1
            assert len(controller._cells) == 1
            assert controller._cells[0].is_alive
        finally:
            await controller.dispose()


class TestControllerRequestConstruction:
    def test_request_captures_the_current_driver_runtime_values(self) -> None:
        """The controller request captures initialized runtime values independently of later driver changes."""
        args = parse_megatron_test_config()
        args.num_rollout = 23
        args.wandb_run_id = "wandb-initialized"
        args.mlflow_run_id = "mlflow-initialized"

        request = TrainerControllerInitRequest.from_args(args, trainer=args.raw_megatron.trainers[0])
        args.num_rollout = 29
        args.wandb_run_id = "wandb-later"
        args.mlflow_run_id = "mlflow-later"

        assert request.num_rollout == 23
        assert request.wandb_run_id == "wandb-initialized"
        assert request.mlflow_run_id == "mlflow-initialized"
