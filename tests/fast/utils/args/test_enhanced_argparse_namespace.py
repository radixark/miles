import torch
from megatron.core.enums import ModelType

from miles.backends.megatron_utils.megatron_config import MegatronConfig


class TestSerializableConfigDict:
    def test_megatron_base_args_survive_a_json_round_trip(self):
        """A served pod rebuilds its config from JSON, so Megatron's dtype and enum base args must come back intact."""
        config = MegatronConfig(
            trainers=[],
            base_args={"params_dtype": torch.bfloat16, "model_type": ModelType.encoder_or_decoder, "seq": (1, 2)},
        )

        restored = MegatronConfig.model_validate_json(config.model_dump_json())

        assert restored.base_args == config.base_args
