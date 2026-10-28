from collections.abc import Callable
from types import SimpleNamespace

import pytest

from miles.backends.training_utils.weight_update.protocols.cuda_ipc import UpdateWeightFromTensor


class TestColocatedLoraBaseResync:
    @pytest.mark.parametrize(
        "backup,checker,offload,needs_resync",
        [
            (True, False, True, False),
            (False, False, True, True),
            (True, True, True, True),
            (False, False, False, False),
        ],
    )
    def test_connect_retains_base_only_when_pause_preserves_it_and_checker_allows_it(
        self,
        colocated_protocol: Callable[..., UpdateWeightFromTensor],
        backup: bool,
        checker: bool,
        offload: bool,
        needs_resync: bool,
    ) -> None:
        """A connected LoRA transfer reuses persisted base weights unless verification forces a rewrite."""
        protocol = colocated_protocol(backup=backup, checker=checker, offload=offload)
        engine = object()

        protocol.connect(
            rollout_engines=[engine],
            engine_gpu_counts=[1],
            engine_gpu_offsets=[0],
            engine_cell_ids=["engine"],
            parallel_state=SimpleNamespace(),
            placement=protocol.required_placement,
            selector="all",
        )

        assert protocol.is_sender is True
        assert protocol.needs_base_resync_for_lora is needs_resync
        assert protocol.rollout_engines == [engine]
