from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.ray.test_placement_group import TestCreateTrainingModels as _TrainingModelsHarness
from tests.fast.ray.test_placement_group import _make_trainer_handle

from miles.ray.placement_group import create_training_models


class TestSinglePolicyCheckpointDispatch:
    @pytest.mark.parametrize("initialized", [False, True])
    async def test_actor_and_critic_receive_their_own_resolved_checkpoint_requests(
        self, tmp_path: Path, initialized: bool
    ) -> None:
        """Initial and resumed orchestration deliver distinct actor and critic paths through actual driver wiring."""
        actor = tmp_path / "actor"
        critic = tmp_path / "critic"
        for directory in (actor, critic):
            directory.mkdir()
            (directory / "latest_checkpointed_iteration.txt").write_text("7")
        args = parse_megatron_test_config(
            "--advantage-estimator", "ppo", "--load", str(actor), "--critic-load", str(critic)
        )
        handles = {role: _make_trainer_handle(initialized=initialized) for role in ("actor", "critic")}
        for handle in handles.values():
            handle.init.return_value = [8]
            handle.load_state.return_value = [8]

        await create_training_models(args, _TrainingModelsHarness._rollout_executor(), handles=handles)

        requests = [
            (handle.load_state if initialized else handle.init).call_args.args[0] for handle in handles.values()
        ]
        assert [request.ckpt_load_config.load for request in requests] == [str(actor), str(critic)]
        assert all(request.ckpt_load_config.resume_from_ckpt for request in requests)
        assert all(not request.ckpt_load_config.no_load_optim for request in requests)
        assert requests[0] is not requests[1]
