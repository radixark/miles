from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

import ray
import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    RemoteWeightLocation,
    query_remote_shards,
)
from miles.backends.training_utils.weight_update.protocols.transports.rollout_engine_write_threads import (
    RolloutEngineWriteThreads,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)


class MooncakeTransport:
    """Writes tensors from this trainer process's registered memory into rollout engines over Mooncake.

    The p2p protocol keeps one for the whole trainer process: it registers its transfer buffers once, calls
    `connect` with each new set of rollout engines, then `write` for each rollout engine rank. Each rollout engine
    has its own write thread, so a stuck engine holds up only its own writes.
    """

    # host memory: no trainer GPU memory, and no GPUDirect RDMA needed
    transfer_buffer_device = torch.device("cpu")

    def __init__(self) -> None:
        self._transfer_engine = _create_transfer_engine()
        self._write_threads = RolloutEngineWriteThreads(thread_name_prefix="mooncake-write")

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        assignments: Sequence[RolloutEngineRankAssignment],
        runner_roles: Sequence[str],
    ) -> dict[str, dict[int, list[RemoteShard]]]:
        """Return assigned shards by runner role and rollout rank."""
        self._write_threads.restart()
        return query_remote_shards(rollout_engines, assignments, runner_roles, _build_remote_shard)

    def register_memory(self, tensor: torch.Tensor) -> None:
        ret = self._transfer_engine.register_memory(tensor.data_ptr(), tensor.nbytes)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake could not register {tensor.nbytes} bytes at {tensor.data_ptr():#x}: error {ret}"
            )

    def unregister_memory(self, tensor: torch.Tensor) -> None:
        ret = self._transfer_engine.unregister_memory(tensor.data_ptr())
        if ret != 0:
            raise RuntimeError(f"Mooncake could not unregister the memory at {tensor.data_ptr():#x}: error {ret}")

    def write(self, remote_shards: Sequence[RemoteShard], tensors_by_name: dict[str, torch.Tensor]) -> list[Future]:
        """Writes each tensor into the weight of the same name on every shard; returns one future per shard.

        The tensors must lie in registered memory and stay unchanged until their futures are done.
        """
        return self._write_threads.write_each_shard(remote_shards, tensors_by_name, self._write_shard)

    def _write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        names = list(tensors_by_name)
        ret = self._transfer_engine.batch_transfer_sync_write(
            remote_shard.transport_peer,
            [tensors_by_name[name].data_ptr() for name in names],
            [remote_shard.weight_locations_by_name[name].address for name in names],
            [tensors_by_name[name].nbytes for name in names],
        )
        if ret < 0:
            raise RuntimeError(f"Mooncake batch_transfer_sync_write returned {ret}")


def _build_remote_shard(
    transfer_engine_info: tuple[str, dict], rollout_engine_ind: int, rollout_engine_rank: int, runner_role: str
) -> RemoteShard:
    session_id, weights_info = transfer_engine_info
    assert (
        session_id is not None
    ), f"the {runner_role} of rollout engine {rollout_engine_ind} rank {rollout_engine_rank} has no Mooncake session"
    return RemoteShard(
        rollout_engine_ind=rollout_engine_ind,
        rollout_engine_rank=rollout_engine_rank,
        runner_role=runner_role,
        transport_peer=session_id,
        weight_locations_by_name={name: RemoteWeightLocation(*location) for name, location in weights_info.items()},
    )


def _create_transfer_engine() -> Any:
    # not in miles' requirements
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    transfer_engine.initialize(ray._private.services.get_node_ip_address(), "P2PHANDSHAKE", "rdma", "")
    return transfer_engine
