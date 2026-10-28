from unittest.mock import patch

import pytest
from tests.fast.backends.training_utils.weight_update.test_broadcast import _BROADCAST_MODULE, _AcceptingEngine

from miles.backends.training_utils.weight_update.protocols.broadcast import connect_rollout_engines_from_distributed


class TestBroadcastCountLength:
    def test_each_engine_requires_a_gpu_count_before_address_lookup_or_group_creation(self) -> None:
        """A count list shorter than the engine fleet fails before any distributed setup."""
        engines = [_AcceptingEngine(), _AcceptingEngine()]

        with (
            patch(f"{_BROADCAST_MODULE}.ray") as ray,
            patch(f"{_BROADCAST_MODULE}.init_process_group") as init_process_group,
        ):
            with pytest.raises(AssertionError):
                connect_rollout_engines_from_distributed(
                    group_name="miles-pp_0", rollout_engines=engines, engine_gpu_counts=[2]
                )

        ray._private.services.get_node_ip_address.assert_not_called()
        init_process_group.assert_not_called()
