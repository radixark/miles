from types import ModuleType
from typing import Any

import pytest
import torch


def test_a_weight_of_another_size_on_the_shard_is_never_written(p2p_sender: Any, mooncake_module: ModuleType) -> None:
    """Writing a weight whose published size differs would run past the target's memory."""
    transport = mooncake_module.MooncakeTransport()
    remote_shard = mooncake_module.RemoteShard(
        rollout_engine_ind=0,
        rollout_engine_rank=0,
        runner_role="target",
        session_id="cell-a-r0",
        weight_locations_by_name={"w": mooncake_module.RemoteWeightLocation(address=0x1000, numel=2, element_size=4)},
    )

    with pytest.raises(
        AssertionError, match="w is 16 bytes here but 8 bytes on the target of rollout engine 0 rank 0"
    ):
        transport.write([remote_shard], {"w": torch.zeros(4)})
    assert p2p_sender.transfer_engine.writes == []
