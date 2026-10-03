from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.runtime import OrchestratorConfig, TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_checkpoint_load, compute_trainer_config
from miles.utils.run_uuid import RUN_UUID_LENGTH


class TestTrainerCheckpointConfiguration:
    def test_checkpoint_creation_preserves_the_complete_trainer_config(self, tmp_path: Path) -> None:
        """Identical launch arguments must keep trainer Pod configuration unchanged after saving."""
        load = tmp_path / "run"
        argv = ("--load", str(load), "--ref-load", str(tmp_path / "reference"), "--run-uuid", "a" * RUN_UUID_LENGTH)
        before = parse_megatron_test_config(*argv)
        [trainer] = before.raw_megatron.trainers
        original = compute_trainer_config(before, trainer).model_dump(mode="json")

        load.mkdir()
        (load / "latest_checkpointed_iteration.txt").write_text("3")
        after = parse_megatron_test_config(*argv)
        [trainer] = after.raw_megatron.trainers
        repeated = compute_trainer_config(after, trainer).model_dump(mode="json")

        assert original == repeated

    @pytest.mark.parametrize("field", ["load", "finetune", "no_load_optim", "no_load_rng", "ckpt_step"])
    def test_runtime_loading_fields_are_rejected_in_trainer_configuration(self, field: str) -> None:
        """A serialized loading decision must not silently become part of a trainer Pod template."""
        args = parse_megatron_test_config()
        [trainer] = args.raw_megatron.trainers
        config = compute_trainer_config(args, trainer)
        assert field not in vars(config.backend)
        values = config.model_dump(mode="python")
        values["backend"] = MegatronArgsNamespace(**(vars(config.backend) | {field: None}))
        with pytest.raises(ValueError, match="Checkpoint load inputs"):
            TrainerConfig.model_validate(values)


class TestOrchestratorCheckpointRequests:
    @pytest.mark.parametrize(
        "extra",
        [(), ("--advantage-estimator", "ppo"), ("--megatron-config", encode_megatron_config("a", "b"))],
    )
    def test_loading_uses_only_fields_owned_by_the_orchestrator(self, extra: tuple[str, ...]) -> None:
        """Checkpoint selection must work after deployment-only scaling fields have been removed."""
        args = parse_megatron_test_config(*extra, "--load", "/run", "--ref-load", "/reference")
        orchestrator = OrchestratorConfig.slice_from(args)

        assert "actor_num_nodes" not in type(orchestrator).model_fields
        for trainer in args.raw_megatron.trainers:
            request = compute_trainer_checkpoint_load(orchestrator, trainer)
            assert request == compute_trainer_checkpoint_load(args, trainer)
            assert request.load == "/reference"
