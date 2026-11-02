import pytest
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.ray.specs.train import TrainerSpec
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.args.trainer_utils import compute_trainer_config


@pytest.mark.parametrize("role,first_workers,second_workers", [("actor", 8, 32), ("critic", 4, 24)])
def test_one_trainer_spec_uses_each_supplied_layout_without_changing_its_worker_payload(
    role: str, first_workers: int, second_workers: int
) -> None:
    """A spec can be scheduled twice at different sizes while its transported worker configuration stays fixed."""
    args = parse_megatron_test_config("--advantage-estimator", "ppo")
    trainer = next(item for item in args.raw_megatron.trainers if item.role == role)
    config = compute_trainer_config(args, trainer)
    spec = TrainerSpec.model_construct(args=config)
    before = config.model_dump_json()
    first = ScalingConfig.model_construct(
        actor_num_nodes=2, actor_num_gpus_per_node=4, critic_num_nodes=1, critic_num_gpus_per_node=4
    )
    second = ScalingConfig.model_construct(
        actor_num_nodes=4, actor_num_gpus_per_node=8, critic_num_nodes=3, critic_num_gpus_per_node=8
    )

    first_schedule = spec.scheduling(first)
    second_schedule = spec.scheduling(second)
    repeated = spec.scheduling(first)

    assert (first_schedule.num_cells, first_schedule.num_workers_per_cell) == (1, first_workers)
    assert (second_schedule.num_cells, second_schedule.num_workers_per_cell) == (1, second_workers)
    assert first_schedule.num_gpus_per_node == 4
    assert second_schedule.num_gpus_per_node == 8
    assert repeated == first_schedule
    assert config.model_dump_json() == before
