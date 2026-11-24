import inspect
from types import ModuleType

import pytest

from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig


class TestTorchtitanCheckpointRequestRejection:
    def test_megatron_load_request_is_rejected_before_torchtitan_initialization(
        self, titan_actor_module: ModuleType
    ) -> None:
        """TorchTitan cannot silently accept Megatron-only runtime loading inputs."""
        actor_type = titan_actor_module.TorchtitanTrainRayActor
        actor = actor_type.__new__(actor_type)
        original = vars(actor).copy()

        with pytest.raises(AssertionError, match="does not support checkpoint healing"):
            inspect.unwrap(actor_type.init)(
                actor,
                args=None,
                role="actor",
                ckpt_load_config=MegatronCkptLoadConfig.for_peer_recovery(),
                indep_dp_info=None,
                indep_dp_store_addr=None,
            )

        assert vars(actor) == original
