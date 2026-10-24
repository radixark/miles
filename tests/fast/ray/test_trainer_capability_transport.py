from unittest.mock import MagicMock

from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.specs.test_train import _make_args

from miles.ray.placement_group import create_trainer_handles
from miles.ray.specs.train import compute_trainer_configs
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


class TestTrainerCapabilityTransport:
    def test_actor_and_critic_handles_use_the_supplied_capability_provider(self) -> None:
        """Every trainer is addressed through the existing capability rather than a recomputed backend."""
        args = _make_args(use_critic=True)
        actor, critic = object(), object()
        provider = MagicMock(spec=BaseWorkerProvider)
        provider.get_handle.side_effect = [actor, critic]
        capability = FakeBackendCapability(static_provider=provider)

        handles = create_trainer_handles(args, trainer_configs=compute_trainer_configs(args), capability=capability)

        assert handles == {"actor": actor, "critic": critic}
        assert capability.requested_static_pool_ids == ["trainer-controller-actor", "trainer-controller-critic"]
        assert [call.args[0] for call in provider.get_handle.call_args_list] == [
            "trainer-controller-actor-00000-00000",
            "trainer-controller-critic-00000-00000",
        ]
