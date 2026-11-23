from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.utils.args.trainer_utils import compute_trainer_config


class TestDefaultActorIndex:
    def test_implicit_single_policy_actor_occupies_the_first_placement_slice(self) -> None:
        """The implicit actor carries placement index zero through the trainer configuration boundary."""
        config = parse_megatron_test_config()
        (actor,) = config.raw_megatron.trainers

        worker_config = compute_trainer_config(config, actor)

        assert actor.actor_index == 0
        assert worker_config.trainer_actor_index == 0
        assert worker_config.trainer_id == "actor"
