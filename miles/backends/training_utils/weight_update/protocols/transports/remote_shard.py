from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

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
    """Published weight addresses of one engine rank's target or draft runner."""

    rollout_engine_ind: int
    rollout_engine_rank: int
    runner_role: str
    # what the transport names the runner it writes into: a Mooncake session or a NIXL agent
    transport_peer: str
    weight_locations_by_name: dict[str, RemoteWeightLocation]

    @property
    def published_nbytes_by_name(self) -> dict[str, int]:
        return {
            name: location.numel * location.element_size for name, location in self.weight_locations_by_name.items()
        }


def query_remote_shards(
    rollout_engines: Sequence[SGLangApiClient],
    assignments: Sequence[RolloutEngineRankAssignment],
    runner_roles: Sequence[str],
    transport: str,
    build_remote_shard: Callable[[dict, int, int, str], RemoteShard],
) -> dict[str, dict[int, list[RemoteShard]]]:
    """Return assigned shards by runner role and rollout rank, each built by `build_remote_shard` from the transfer
    engine info its rollout engine rank published for `transport`."""
    return {
        runner_role: {
            assignment.rollout_engine_rank: [
                build_remote_shard(
                    _query_transfer_engine_info(
                        rollout_engines[rollout_engine_ind],
                        rollout_engine_ind,
                        assignment.rollout_engine_rank,
                        runner_role,
                        transport,
                    ),
                    rollout_engine_ind,
                    assignment.rollout_engine_rank,
                    runner_role,
                )
                for rollout_engine_ind in assignment.rollout_engine_indices
            ]
            for assignment in assignments
        }
        for runner_role in runner_roles
    }


def parse_weight_locations(transfer_engine_info: dict) -> dict[str, RemoteWeightLocation]:
    return {
        name: RemoteWeightLocation(*location) for name, location in transfer_engine_info["weights_info_dict"].items()
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


def _query_transfer_engine_info(
    rollout_engine: SGLangApiClient,
    rollout_engine_ind: int,
    rollout_engine_rank: int,
    runner_role: str,
    transport: str,
) -> dict:
    transfer_engine_info = async_utils.run(
        rollout_engine.get_remote_instance_transfer_engine_info(rank=rollout_engine_rank, role=runner_role)
    )
    assert transfer_engine_info["transport"] == transport, (
        f"the {runner_role} of rollout engine {rollout_engine_ind} rank {rollout_engine_rank} publishes its weights "
        f"for {transfer_engine_info['transport']}, but --p2p-transport is {transport}"
    )
    return transfer_engine_info
