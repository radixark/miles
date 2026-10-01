import pytest

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace, MegatronConfig
from miles.utils.args.runtime import AllConfig, TrainerConfig
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp


@pytest.mark.parametrize("config_type", [AllConfig, TrainerConfig])
def test_parallel_properties_read_backend_values_without_duplicate_fields(config_type: type) -> None:
    """Read parallel dimensions from their owner without creating writable duplicate fields."""
    dimensions = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 3,
        "context_parallel_size": 4,
    }
    config = (
        AllConfig.model_construct(
            train_backend="megatron", raw_megatron=MegatronConfig(trainers=[], base_args=dimensions)
        )
        if config_type is AllConfig
        else TrainerConfig.model_construct(backend=MegatronArgsNamespace(**dimensions))
    )

    assert compute_megatron_world_size_except_dp(config) == 24
    assert not dimensions.keys() & config_type.model_fields.keys()
    with pytest.raises((AttributeError, ValueError, TypeError)):
        config.tensor_model_parallel_size = 8
