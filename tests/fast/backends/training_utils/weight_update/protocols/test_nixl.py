from types import ModuleType
from typing import Any

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks_for_data_replica,
)


@pytest.fixture
def p2p_transport() -> str:
    return "nixl"


def test_a_write_addresses_the_gpu_its_rollout_engine_rank_published(
    p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
) -> None:
    """A NIXL write names the remote GPU; another rank's would write into the wrong device's memory."""
    protocol = p2p_sender.make_protocol()
    api = make_rollout_api("cell-a", gpu_count=2)
    p2p_sender.connect(protocol, [api])
    p2p_sender.begin_sync(protocol, weight_version=1)

    protocol.send_bucket(make_bucket("hf.w"))
    protocol.after_base_weights()

    assert p2p_sender.nixl_agent.remote_gpu_ids_by_remote_agent == {api.session_id(0): {0}, api.session_id(1): {1}}


def test_a_transfer_still_running_after_the_timeout_fails_its_write(
    p2p_sender: Any, p2p_protocol: ModuleType, make_rollout_api: Any, make_cell_updater: Any
) -> None:
    """NIXL never abandons a transfer itself: a lost rollout engine would hold its write thread, and every later
    write to that engine, forever."""
    transport = p2p_protocol.P2PTransport(p2p_protocol.NixlTransport(write_timeout=0.05))
    api = make_rollout_api("cell-a", gpu_count=1)
    assignments = assign_rollout_engine_ranks_for_data_replica(
        data_replica_rank=0, data_replica_size=1, engine_gpu_counts=[1]
    )
    (remote_shard,) = transport.connect(
        {"cell-a": make_cell_updater("cell-a", api_client=api)}, ["cell-a"], assignments, ["target"]
    )["target"][0]
    buffer = torch.zeros(4)
    transport.register_memory(buffer)
    p2p_sender.nixl_agent.stuck_remote_agents.add(api.session_id(0))

    (write,) = transport.write([remote_shard], {"w": buffer}, run_write=lambda shard, write: write())

    with pytest.raises(RuntimeError, match=r"NIXL WRITE of 1 weights is PROC after 0.05s"):
        write.result(timeout=5)


def test_a_reconnect_removes_the_agents_of_the_rollout_engines_it_replaces(
    p2p_sender: Any, make_rollout_api: Any
) -> None:
    """Agents left loaded for replaced engines would pile up, one set per rollout engine restart."""
    protocol = p2p_sender.make_protocol()
    p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])

    replaced_api = make_rollout_api("cell-a", gpu_count=1, generation=2)
    p2p_sender.connect(protocol, [replaced_api])

    assert p2p_sender.nixl_agent.remote_agents == [replaced_api.session_id(0)]


def test_a_reconnect_fails_a_write_still_running_to_a_replaced_agent_and_keeps_its_memory(
    p2p_sender: Any, p2p_protocol: ModuleType, make_rollout_api: Any, make_cell_updater: Any
) -> None:
    """NIXL fails a transfer whose remote agent a reconnect removed, without saying whether it read its source."""
    transport = p2p_protocol.P2PTransport(p2p_protocol.NixlTransport(write_timeout=10))
    assignments = assign_rollout_engine_ranks_for_data_replica(
        data_replica_rank=0, data_replica_size=1, engine_gpu_counts=[1]
    )
    api = make_rollout_api("cell-a", gpu_count=1)
    (remote_shard,) = transport.connect(
        {"cell-a": make_cell_updater("cell-a", api_client=api)}, ["cell-a"], assignments, ["target"]
    )["target"][0]
    buffer = torch.zeros(4)
    transport.register_memory(buffer)
    p2p_sender.nixl_agent.stuck_remote_agents.add(api.session_id(0))
    (write,) = transport.write([remote_shard], {"w": buffer}, run_write=lambda shard, write: write())
    assert p2p_sender.nixl_agent.transfer_started.wait(timeout=5)

    replaced_api = make_rollout_api("cell-a", gpu_count=1, generation=2)
    transport.connect(
        {"cell-a": make_cell_updater("cell-a", api_client=replaced_api)}, ["cell-a"], assignments, ["target"]
    )

    with pytest.raises(RuntimeError, match="NIXL_ERR_NOT_FOUND"):
        write.result(timeout=5)
    with pytest.raises(RuntimeError, match="unknown native completion"):
        transport.unregister_memory(buffer)
