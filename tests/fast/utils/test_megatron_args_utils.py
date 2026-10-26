from types import SimpleNamespace

import pytest

from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.trainer_utils import compute_trainer_total_gpus
from miles.utils.megatron_args_utils import compute_trainer_num_cells


@pytest.mark.parametrize("role,total_gpus,independent_cells", [("actor", 16, 2), ("critic", 24, 3)])
@pytest.mark.parametrize("independent_dp", [False, True])
def test_independent_cells_include_tensor_pipeline_and_context_parallelism(
    role: str, total_gpus: int, independent_cells: int, independent_dp: bool
) -> None:
    """Every model-parallel dimension belongs to a cell, using each trainer role's GPU count."""
    scaling = ScalingConfig.model_construct(
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        critic_num_nodes=3,
        critic_num_gpus_per_node=8,
    )
    args = SimpleNamespace(
        indep_dp=independent_dp,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
    )

    actual_gpus = compute_trainer_total_gpus(scaling, role=role)
    actual_cells = compute_trainer_num_cells(args, total_gpus=actual_gpus)

    assert actual_gpus == total_gpus
    assert actual_cells == (independent_cells if independent_dp else 1)
