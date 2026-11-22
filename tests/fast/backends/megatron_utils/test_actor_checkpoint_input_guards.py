import inspect
from types import ModuleType

import pytest
from tests.fast.backends.megatron_utils.test_actor import actor_module

from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig

__all__ = ["actor_module"]


class TestActorCheckpointInputGuards:
    @pytest.mark.parametrize("peer_recovery", [False, True])
    def test_missing_startup_request_and_conflicting_peer_request_fail_before_initialization(
        self, actor_module: ModuleType, peer_recovery: bool
    ) -> None:
        """Initialization rejects absent startup inputs and disk inputs attached to peer recovery."""
        actor_type = actor_module.MegatronTrainRayActor
        actor = actor_type.__new__(actor_type)
        original = vars(actor).copy()
        request = MegatronCkptLoadConfig.for_peer_recovery() if peer_recovery else None
        error = "Peer recovery does not use disk" if peer_recovery else "initialization requires checkpoint"

        with pytest.raises(AssertionError, match=error):
            inspect.unwrap(actor_type.init)(
                actor,
                args=None,
                role="actor",
                ckpt_load_config=request,
                recv_ckpt_src_rank=0 if peer_recovery else None,
                indep_dp_info=None,
                indep_dp_store_addr=None,
            )

        assert vars(actor) == original
