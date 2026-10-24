import pytest

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.runtime import TrainerConfig
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp


def test_parallel_properties_read_backend_values_without_duplicate_fields() -> None:
    """Read parallel dimensions from their owner without creating writable duplicate fields."""
    dimensions = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 3,
        "context_parallel_size": 4,
    }
    config = TrainerConfig.model_construct(backend=MegatronArgsNamespace(**dimensions))

    assert compute_megatron_world_size_except_dp(config) == 24
    assert not dimensions.keys() & TrainerConfig.model_fields.keys()
    with pytest.raises((AttributeError, ValueError, TypeError)):
        config.tensor_model_parallel_size = 8
