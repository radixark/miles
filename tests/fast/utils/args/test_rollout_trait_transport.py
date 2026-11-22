import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.utils.args.configs.algo import AlgoRolloutOnlyConfig
from miles.utils.args.configs.ci import CiRolloutOnlyConfig
from miles.utils.args.configs.debug import DebugRolloutOnlyConfig
from miles.utils.args.configs.eval import EvalRolloutOnlyConfig
from miles.utils.args.configs.on_policy_distillation import OnPolicyDistillationRolloutOnlyConfig
from miles.utils.args.configs.reward_model import RewardModelRolloutOnlyConfig
from miles.utils.args.configs.rollout import RolloutRelatedRolloutOnlyConfig
from miles.utils.args.configs.rollout_buffer import RolloutBufferRolloutOnlyConfig
from miles.utils.args.configs.train import TrainRolloutOnlyConfig
from miles.utils.args.configs.wandb import WandbRolloutOnlyConfig
from miles.utils.args.runtime import InferenceControllerConfig, OrchestratorConfig, RolloutConfig
from miles.utils.args.schema import BaseConfig
from miles.utils.args.trainer_utils import compute_trainer_config

_ROLLOUT_ONLY_TRAITS = [
    AlgoRolloutOnlyConfig,
    CiRolloutOnlyConfig,
    DebugRolloutOnlyConfig,
    EvalRolloutOnlyConfig,
    OnPolicyDistillationRolloutOnlyConfig,
    RewardModelRolloutOnlyConfig,
    RolloutRelatedRolloutOnlyConfig,
    RolloutBufferRolloutOnlyConfig,
    TrainRolloutOnlyConfig,
    WandbRolloutOnlyConfig,
]


class TestRolloutTraitTransport:
    @pytest.mark.parametrize("trait", _ROLLOUT_ONLY_TRAITS, ids=lambda trait: trait.__name__)
    def test_each_rollout_only_trait_survives_its_owner_payload_without_leaking_to_other_roles(
        self, trait: type[BaseConfig]
    ) -> None:
        """Role slicing must retain every rollout-only field in exactly its owning payload."""
        config = parse_megatron_test_config(
            "--save-debug-rollout-data",
            "/tmp/debug-{rollout_id}.pt",
            "--rollout-temperature",
            "0.37",
            "--disable-rewards-normalization",
        )
        original = config.model_dump(mode="json")
        rollout = RolloutConfig.slice_from(config)
        restored = RolloutConfig.model_validate_json(rollout.model_dump_json()).model_dump(mode="json")
        others = [
            OrchestratorConfig.slice_from(config),
            InferenceControllerConfig.slice_from(config),
            compute_trainer_config(all_config=config, trainer=config.raw_megatron.trainers[0]),
        ]

        assert trait.model_fields
        for name in trait.model_fields:
            assert name in restored
            assert restored[name] == original[name]
        for component in others:
            assert not trait.model_fields.keys() & component.model_dump(mode="json").keys()
