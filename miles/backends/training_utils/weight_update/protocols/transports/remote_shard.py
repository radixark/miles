from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater


class RemoteWeightLocation(NamedTuple):
    address: int
    numel: int
    element_size: int


@dataclass(frozen=True)
class RemoteShard:
    """Published weight addresses of one engine rank's target or draft runner."""

    cell_id: str
    rollout_engine_ind: int
    rollout_engine_rank: int
    runner_role: str
    # what the transport names the runner it writes into, such as a Mooncake session or a NIXL agent
    transport_peer: str
    weight_locations_by_name: dict[str, RemoteWeightLocation]

    @property
    def published_nbytes_by_name(self) -> dict[str, int]:
        return {
            name: location.numel * location.element_size for name, location in self.weight_locations_by_name.items()
        }


def query_remote_shards(
    cell_updaters_of_cell_id: dict[str, _RolloutCellUpdater],
    engine_cell_ids: Sequence[str],
    assignments: Sequence[RolloutEngineRankAssignment],
    runner_roles: Sequence[str],
    transport_name: str,
    connect_peer: Callable[[dict], str],
) -> dict[str, dict[int, list[RemoteShard]]]:
    """Return assigned shards by runner role and rollout rank, built from the transfer engine info each rollout engine
    rank published for `transport_name`; `connect_peer` turns that info into the shard's transport peer. A rank whose
    cell returned no info is left out."""
    return {
        runner_role: {
            assignment.rollout_engine_rank: [
                remote_shard
                for rollout_engine_ind in assignment.rollout_engine_indices
                if (
                    remote_shard := _query_remote_shard(
                        cell_updaters_of_cell_id[engine_cell_ids[rollout_engine_ind]],
                        rollout_engine_ind,
                        assignment.rollout_engine_rank,
                        runner_role,
                        transport_name,
                        connect_peer,
                    )
                )
                is not None
            ]
            for assignment in assignments
        }
        for runner_role in runner_roles
    }


def assert_tensors_fit(remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
    target = (
        f"the {remote_shard.runner_role} of rollout engine {remote_shard.rollout_engine_ind} "
        f"rank {remote_shard.rollout_engine_rank}"
    )
    for name, tensor in tensors_by_name.items():
        assert name in remote_shard.weight_locations_by_name, f"{target} publishes no {name}"
        location = remote_shard.weight_locations_by_name[name]
        target_nbytes = location.numel * location.element_size
        assert target_nbytes == tensor.nbytes, (
            f"{name} is {tensor.nbytes} bytes here but {target_nbytes} bytes on {target}; "
            "writing it would run past the target"
        )


def _query_remote_shard(
    cell_updater: _RolloutCellUpdater,
    rollout_engine_ind: int,
    rollout_engine_rank: int,
    runner_role: str,
    transport_name: str,
    connect_peer: Callable[[dict], str],
) -> RemoteShard | None:
    transfer_engine_info = cell_updater.submit_client_call(
        "get_remote_instance_transfer_engine_info", rank=rollout_engine_rank, role=runner_role
    ).result()
    if transfer_engine_info is None:
        return None
    assert transfer_engine_info["transport"] == transport_name, (
        f"the {runner_role} of rollout engine {rollout_engine_ind} rank {rollout_engine_rank} publishes its weights "
        f"for {transfer_engine_info['transport']}, but --p2p-transport is {transport_name}"
    )
    return RemoteShard(
        cell_id=cell_updater.cell_id,
        rollout_engine_ind=rollout_engine_ind,
        rollout_engine_rank=rollout_engine_rank,
        runner_role=runner_role,
        transport_peer=connect_peer(transfer_engine_info),
        weight_locations_by_name={
            name: RemoteWeightLocation(*location)
            for name, location in transfer_engine_info["weights_info_dict"].items()
        },
    )
