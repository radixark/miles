from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, NamedTuple

import ray
import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.utils import async_utils


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

    The p2p protocol creates one at each connect: it calls `connect` with the rollout engines, registers its
    source tensors, then calls `write` for each rollout engine rank.
    """

    def __init__(self, num_write_workers: int) -> None:
        self._transfer_engine = _create_transfer_engine()
        self._write_executor = ThreadPoolExecutor(max_workers=num_write_workers)

    def connect(
        self, rollout_engines: Sequence[SGLangApiClient], assignments: Sequence[RolloutEngineRankAssignment]
    ) -> dict[int, list[RemoteShard]]:
        """Returns the shards of the rollout engine ranks in `assignments`, by rollout engine rank."""
        return {
            assignment.rollout_engine_rank: [
                _query_remote_shard(
                    rollout_engines[rollout_engine_ind], rollout_engine_ind, assignment.rollout_engine_rank
                )
                for rollout_engine_ind in assignment.rollout_engine_indices
            ]
            for assignment in assignments
        }

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
            self._write_executor.submit(self._write_shard, remote_shard, tensors_by_name)
            for remote_shard in remote_shards
        ]

    def _write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        names = list(tensors_by_name)
        target_addresses = [
            remote_shard.weight_locations_by_name[name].address
            for name in names
            if name in remote_shard.weight_locations_by_name
        ]
        assert len(target_addresses) == len(names), (
            f"[P2P-Shared] Pointer count mismatch for session {remote_shard.session_id}, "
            f"source: {len(names)}, target: {len(target_addresses)}"
        )
        ret = self._transfer_engine.batch_transfer_sync_write(
            remote_shard.session_id,
            [tensors_by_name[name].data_ptr() for name in names],
            target_addresses,
            [_nbytes(tensors_by_name[name]) for name in names],
        )
        if ret < 0:
            raise RuntimeError(f"[P2P-Shared] Transfer failed for session {remote_shard.session_id}, error: {ret}")


def _query_remote_shard(
    rollout_engine: SGLangApiClient, rollout_engine_ind: int, rollout_engine_rank: int
) -> RemoteShard:
    session_id, weights_info = async_utils.run(
        rollout_engine.get_remote_instance_transfer_engine_info(rank=rollout_engine_rank)
    )
    assert (
        session_id is not None
    ), f"rollout engine {rollout_engine_ind} rank {rollout_engine_rank} has no Mooncake session"
    return RemoteShard(
        rollout_engine_ind=rollout_engine_ind,
        rollout_engine_rank=rollout_engine_rank,
        session_id=session_id,
        weight_locations_by_name={name: RemoteWeightLocation(*location) for name, location in weights_info.items()},
    )


def _create_transfer_engine() -> Any:
    # not in miles' requirements
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    transfer_engine.initialize(ray._private.services.get_node_ip_address(), "P2PHANDSHAKE", "rdma", "")
    return transfer_engine


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
