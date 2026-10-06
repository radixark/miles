from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, NamedTuple

import ray
import torch


class RemoteWeightLocation(NamedTuple):
    address: int
    numel: int
    element_size: int


@dataclass(frozen=True)
class RemoteShard:
    """One rank of one rollout engine as a write target: its Mooncake session and where each weight lives."""

    rollout_engine_ind: int
    rollout_engine_rank: int
    session_id: str
    weight_locations_by_name: dict[str, RemoteWeightLocation]


class MooncakeTransport:
    """Writes tensors from this trainer process's registered memory into rollout engines over Mooncake.

    The p2p protocol registers its source tensors once, then calls `write` for each rollout engine rank. Each
    rollout engine has its own write thread, so a stuck engine holds up only its own writes.
    """

    def __init__(self) -> None:
        self._transfer_engine = _create_transfer_engine()
        self._write_executors_by_rollout_engine_ind: dict[int, ThreadPoolExecutor] = {}

    def register_memory(self, tensor: torch.Tensor) -> None:
        ret = self._transfer_engine.register_memory(tensor.data_ptr(), _nbytes(tensor))
        if ret != 0:
            raise RuntimeError(
                f"Mooncake could not register {_nbytes(tensor)} bytes at {tensor.data_ptr():#x}: error {ret}"
            )

    def write(self, remote_shards: Sequence[RemoteShard], tensors_by_name: dict[str, torch.Tensor]) -> list[Future]:
        """Writes each tensor into the weight of the same name on every shard; returns one future per shard.

        The tensors must lie in registered memory and stay unchanged until their futures are done.
        """
        return [
            self._write_executor(remote_shard.rollout_engine_ind).submit(
                self._write_shard, remote_shard, tensors_by_name
            )
            for remote_shard in remote_shards
        ]

    def _write_executor(self, rollout_engine_ind: int) -> ThreadPoolExecutor:
        if rollout_engine_ind not in self._write_executors_by_rollout_engine_ind:
            self._write_executors_by_rollout_engine_ind[rollout_engine_ind] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mooncake-write-rollout-engine-{rollout_engine_ind}"
            )
        return self._write_executors_by_rollout_engine_ind[rollout_engine_ind]

    def _write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        names = list(tensors_by_name)
        ret = self._transfer_engine.batch_transfer_sync_write(
            remote_shard.session_id,
            [tensors_by_name[name].data_ptr() for name in names],
            [remote_shard.weight_locations_by_name[name].address for name in names],
            [_nbytes(tensors_by_name[name]) for name in names],
        )
        if ret < 0:
            raise RuntimeError(f"Mooncake batch_transfer_sync_write returned {ret}")


def _create_transfer_engine() -> Any:
    # not in miles' requirements
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    transfer_engine.initialize(ray._private.services.get_node_ip_address(), "P2PHANDSHAKE", "rdma", "")
    return transfer_engine


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
