from pathlib import Path

from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.fixtures.megatron_config_fixtures import write_megatron_config_trainers
from tests.fast.ray.test_placement_group import TestCreateTrainingModels as _TrainingModelsHarness
from tests.fast.ray.test_placement_group import _make_trainer_handle

from miles.utils.multi_policy.utils import create_trainers


class TestMultiPolicyCheckpointDispatch:
    async def test_each_policy_receives_its_own_initial_weight_source(self, tmp_path: Path) -> None:
        """Each policy's initial request retains its own HF override through real multi-policy driver wiring."""
        config_path = write_megatron_config_trainers(
            tmp_path,
            [
                {"model_id": "a", "overrides": {"hf_checkpoint": "/models/a"}},
                {"model_id": "b", "overrides": {"hf_checkpoint": "/models/b"}},
            ],
        )
        args = parse_megatron_test_config("--megatron-config", config_path, "--megatron-to-hf-mode", "bridge")
        handles = {trainer_id: _make_trainer_handle() for trainer_id in ("a-actor", "b-actor")}

        trainers = await create_trainers(
            args, rollout_executor=_TrainingModelsHarness._rollout_executor(), handles=handles
        )

        requests = [handle.init.call_args.args[0] for handle in handles.values()]
        assert [request.ckpt_load_config.load for request in requests] == ["/models/a", "/models/b"]
        assert all(request.ckpt_load_config.resume_from_ckpt is False for request in requests)
        assert requests[0] is not requests[1]
        assert set(trainers) == {"a", "b"}
