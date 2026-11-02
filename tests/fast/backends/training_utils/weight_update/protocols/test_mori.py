import base64
import gc
import sys
from types import ModuleType
from typing import Any

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    RemoteWeightLocation,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks_for_data_replica,
)

_ENGINE_KEY = "sglang-weights-tp0"
_BLOCK_ADDRESS = 0x10000


@pytest.fixture
def p2p_transport() -> str:
    return "mori"


def _block(address: int, nbytes: int) -> list:
    packed = base64.b64encode(f"{_ENGINE_KEY}|{address}|{nbytes}".encode()).decode("ascii")
    return [address, nbytes, packed]


def _info(blocks: list[list]) -> dict:
    return {
        "transport": "mori",
        "engine_desc": base64.b64encode(_ENGINE_KEY.encode()).decode("ascii"),
        "memory_blocks": blocks,
        "weights_info_dict": {},
    }


def _shard(locations: dict[str, int], numel: int = 4) -> RemoteShard:
    return RemoteShard(
        cell_id="cell-a",
        rollout_engine_ind=0,
        rollout_engine_rank=0,
        runner_role="target",
        transport_peer=_ENGINE_KEY,
        weight_locations_by_name={
            name: RemoteWeightLocation(address=address, numel=numel, element_size=4)
            for name, address in locations.items()
        },
    )


@pytest.fixture
def mori(p2p_sender: Any, p2p_protocol: ModuleType) -> Any:
    return p2p_protocol.MoriTransport(write_timeout=5, transfer_buffer="cpu")


@pytest.fixture
def platform(p2p_protocol: ModuleType) -> Any:
    return sys.modules[p2p_protocol.MoriTransport.__module__].current_platform


def test_weights_in_one_published_block_go_in_one_batch_entry(p2p_sender: Any, mori: Any) -> None:
    """sglang registers the allocator blocks holding the weights, merged where contiguous: weights sharing a block
    share its MemoryDesc, so they are one batch entry of several (offset, size) pairs, not one entry each."""
    mori.connect_peer(_info([_block(_BLOCK_ADDRESS, 64)]))
    buffer = torch.arange(8, dtype=torch.float32)
    mori.register_tensor(buffer)

    mori.write_shard(_shard({"a": _BLOCK_ADDRESS + 32, "b": _BLOCK_ADDRESS}), {"a": buffer[0:4], "b": buffer[4:8]})

    assert p2p_sender.mori_engine.batch_writes == [(_ENGINE_KEY, 1)]
    assert p2p_sender.transfer_engine.payload_of(_ENGINE_KEY) == {
        _BLOCK_ADDRESS + 32: [0.0, 1.0, 2.0, 3.0],
        _BLOCK_ADDRESS: [4.0, 5.0, 6.0, 7.0],
    }


def test_weights_in_different_blocks_go_in_one_entry_per_block(p2p_sender: Any, mori: Any) -> None:
    """A MemoryDesc names one registered block, so a batch entry can only address bytes inside it."""
    mori.connect_peer(_info([_block(_BLOCK_ADDRESS, 16), _block(_BLOCK_ADDRESS + 4096, 16)]))
    buffer = torch.arange(8, dtype=torch.float32)
    mori.register_tensor(buffer)

    mori.write_shard(_shard({"a": _BLOCK_ADDRESS, "b": _BLOCK_ADDRESS + 4096}), {"a": buffer[0:4], "b": buffer[4:8]})

    assert p2p_sender.mori_engine.batch_writes == [(_ENGINE_KEY, 2)]


def test_a_weight_outside_every_published_block_fails_before_anything_is_written(p2p_sender: Any, mori: Any) -> None:
    """A weight sglang did not register (a view into another allocation, say) has no MemoryDesc to address it by."""
    mori.connect_peer(_info([_block(_BLOCK_ADDRESS, 16)]))
    buffer = torch.arange(8, dtype=torch.float32)
    mori.register_tensor(buffer)

    with pytest.raises(RuntimeError, match=r"b \(16 bytes at 0x10008\) is not in any weight block"):
        mori.write_shard(_shard({"a": _BLOCK_ADDRESS, "b": _BLOCK_ADDRESS + 8}), {"a": buffer[0:4], "b": buffer[4:8]})
    assert p2p_sender.mori_engine.batch_writes == []


def test_a_source_outside_registered_memory_fails_before_anything_is_written(p2p_sender: Any, mori: Any) -> None:
    mori.connect_peer(_info([_block(_BLOCK_ADDRESS, 16)]))

    with pytest.raises(RuntimeError, match="is not in memory registered with MoRI"):
        mori.write_shard(_shard({"a": _BLOCK_ADDRESS}), {"a": torch.zeros(4)})
    assert p2p_sender.mori_engine.batch_writes == []


def test_a_host_transfer_buffer_registers_as_cpu_memory(p2p_protocol: ModuleType, mori: Any) -> None:
    """MoRI needs the location to pick ibv_reg_mr over dma-buf registration."""
    region = mori.register_tensor(torch.zeros(4))

    assert p2p_protocol.MoriTransport.transfer_buffer_device == torch.device("cpu")
    assert (region.memory_desc.location, region.memory_desc.device) == ("CPU", -1)


def test_a_transfer_still_running_after_the_timeout_fails_its_write(
    p2p_sender: Any, p2p_protocol: ModuleType, make_rollout_api: Any, make_cell_updater: Any
) -> None:
    """A lost rollout engine would otherwise hold its write thread, and every later write to that engine, forever."""
    transport = p2p_protocol.P2PTransport(p2p_protocol.MoriTransport(write_timeout=0.05, transfer_buffer="cpu"))
    api = make_rollout_api("cell-a", gpu_count=1)
    assignments = assign_rollout_engine_ranks_for_data_replica(
        data_replica_rank=0, data_replica_size=1, engine_gpu_counts=[1]
    )
    (remote_shard,) = transport.connect(
        {"cell-a": make_cell_updater("cell-a", api_client=api)}, ["cell-a"], assignments, ["target"]
    )["target"][0]
    buffer = torch.zeros(4)
    transport.register_memory(buffer)
    p2p_sender.mori_engine.stuck_remote_engines.add(api.session_id(0))

    (write,) = transport.write([remote_shard], {"w": buffer}, run_write=lambda shard, write: write())

    with pytest.raises(
        RuntimeError, match=r"MoRI WRITE of 1 weights .* 1 of 1 batch entries still running after 0.05s"
    ):
        write.result(timeout=5)
    with pytest.raises(RuntimeError, match="unknown native completion"):
        transport.unregister_memory(buffer)


def test_a_reconnect_deregisters_the_engines_of_the_rollout_engines_it_replaces(
    p2p_sender: Any, make_rollout_api: Any
) -> None:
    """Engines left registered for replaced rollout engines would pile up, one set per rollout engine restart."""
    protocol = p2p_sender.make_protocol()
    p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])

    replaced_api = make_rollout_api("cell-a", gpu_count=1, generation=2)
    p2p_sender.connect(protocol, [replaced_api])

    assert p2p_sender.mori_engine.remote_engines == [replaced_api.session_id(0)]


def test_a_write_still_running_at_its_timeout_keeps_its_statuses(p2p_sender: Any, p2p_protocol: ModuleType) -> None:
    """MoRI completes a transfer's status through a raw pointer, also after the write gave up on it."""
    mori = p2p_protocol.MoriTransport(write_timeout=0.05, transfer_buffer="cpu")
    mori.connect_peer(_info([_block(_BLOCK_ADDRESS, 16)]))
    buffer = torch.zeros(4)
    mori.register_tensor(buffer)
    p2p_sender.mori_engine.stuck_remote_engines.add(_ENGINE_KEY)

    with pytest.raises(RuntimeError, match="still running"):
        mori.write_shard(_shard({"a": _BLOCK_ADDRESS}), {"a": buffer})
    gc.collect()

    assert [status() is not None for status in p2p_sender.mori_engine.stuck_statuses] == [True]


def test_gpu_transfer_buffers_refuse_expandable_segment_memory(
    p2p_sender: Any, p2p_protocol: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MoRI cannot register VMM memory; failing at construction names the fix instead of a failed registration."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with pytest.raises(ValueError, match="--p2p-mori-transfer-buffer cpu"):
        p2p_protocol.MoriTransport(write_timeout=5, transfer_buffer="gpu")
    assert p2p_sender.transfer_engines_created == 0


def test_a_gpu_transfer_buffer_is_read_only_after_its_loads_ran(
    p2p_sender: Any, p2p_protocol: ModuleType, platform: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loads into a GPU buffer are queued kernels; the NIC would otherwise read bytes they have yet to write."""
    order: list[str] = []
    monkeypatch.setattr(platform, "current_device", lambda: torch.device("cuda", 3))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda device=None: order.append(f"synchronize {device}"))
    batch_write = p2p_sender.mori_engine.batch_write
    monkeypatch.setattr(
        p2p_sender.mori_engine, "batch_write", lambda *args: order.append("batch_write") or batch_write(*args)
    )
    mori = p2p_protocol.MoriTransport(write_timeout=5, transfer_buffer="gpu")
    mori.connect_peer(_info([_block(_BLOCK_ADDRESS, 16)]))
    buffer = torch.zeros(4)
    mori.register_tensor(buffer)

    mori.write_shard(_shard({"a": _BLOCK_ADDRESS}), {"a": buffer})

    assert mori.transfer_buffer_device == torch.device("cuda", 3)
    assert order == ["synchronize cuda:3", "batch_write"]
