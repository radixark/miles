from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

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
    # what the transport names the runner it writes into, such as a Mooncake session
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
    build_remote_shard: Callable[[Any, int, int, str], RemoteShard],
) -> dict[str, dict[int, list[RemoteShard]]]:
    """Return assigned shards by runner role and rollout rank, each built by `build_remote_shard` from the transfer
    engine info its rollout engine rank published."""
    return {
        runner_role: {
            assignment.rollout_engine_rank: [
                build_remote_shard(
                    async_utils.run(
                        rollout_engines[rollout_engine_ind].get_remote_instance_transfer_engine_info(
                            rank=assignment.rollout_engine_rank, role=runner_role
                        )
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
