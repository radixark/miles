import inspect

import pytest

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig


class TestFsdpCheckpointRequestRejection:
    def test_megatron_load_request_is_rejected_before_fsdp_initialization(self) -> None:
        """FSDP cannot silently accept Megatron-only runtime loading inputs."""
        actor = FSDPTrainRayActor.__new__(FSDPTrainRayActor)
        original = vars(actor).copy()

        with pytest.raises(AssertionError):
            inspect.unwrap(FSDPTrainRayActor.init)(
                actor,
                args=None,
                role="actor",
                ckpt_load_config=MegatronCkptLoadConfig.for_peer_recovery(),
                indep_dp_info=None,
                indep_dp_store_addr=None,
            )

        assert vars(actor) == original
