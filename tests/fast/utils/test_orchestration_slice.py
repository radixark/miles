from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.args.runtime import OrchestratorConfig
from miles.utils.orchestration_utils import ArgvOrchestratorStartupInfo


class TestArgvOrchestratorStartupInfo:
    def test_the_driver_slice_excludes_launcher_fields_and_keeps_the_full_config_for_worker_startup(self) -> None:
        """The driver owns a separate runtime slice while worker startup retains launcher inputs."""
        all_args = parse_fsdp_test_config("--rollout-num-gpus", "4")

        startup = ArgvOrchestratorStartupInfo.create(all_args)

        assert startup.all_args is all_args
        assert isinstance(startup.args, OrchestratorConfig)
        assert startup.args is not all_args
        assert startup.args.num_rollout == all_args.num_rollout
        assert "rollout_num_gpus" not in startup.args.model_dump()
        assert "sglang_scaling" not in startup.args.model_dump()
        assert all_args.rollout_num_gpus == 4
        startup.args.num_rollout = 7
        startup.args.wandb_run_id = "driver-generated"
        assert all_args.num_rollout == 1
        assert all_args.wandb_run_id is None
