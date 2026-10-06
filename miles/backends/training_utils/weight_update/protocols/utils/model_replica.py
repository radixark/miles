from collections.abc import Sequence
from dataclasses import dataclass

from sglang.srt.distributed.parallel_state import RankParallelismConfig
from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.utils import async_utils
from miles.utils.workers.argv_utils import _record_field_names

# where a rank sits in the launch, not how it holds its weights
_PLACEMENT_PARALLELISM_FIELDS = frozenset({"global_rank", "local_rank"})


@dataclass(frozen=True)
class RolloutEngineRankConfig:
    """How one rollout engine rank holds its weights, as the engine reports it.

    Ranks with the same `shard_layout_key` take the same bytes, so one model replica serves them all.
    """

    parallelism: RankParallelismConfig
    server_args: ServerArgs

    @property
    def shard_layout_key(self) -> tuple:
        sharding = {
            name: value
            for name, value in self.parallelism.to_dict().items()
            if name not in _PLACEMENT_PARALLELISM_FIELDS
        }
        return tuple(sorted(sharding.items())), self.server_args.quantization


def query_rollout_engine_rank_configs(
    rollout_engines: Sequence[SGLangApiClient], assignments: Sequence[RolloutEngineRankAssignment]
) -> dict[int, RolloutEngineRankConfig]:
    """Returns the config of each rollout engine rank in `assignments`, by rollout engine rank.

    All rollout engines of one rank must hold it the same way, since one model replica serves them.
    """
    configs_by_rollout_engine_rank = {}
    for assignment in assignments:
        configs = [
            _query_config(rollout_engines[rollout_engine_ind], assignment.rollout_engine_rank)
            for rollout_engine_ind in assignment.rollout_engine_indices
        ]
        shard_layout_keys = {config.shard_layout_key for config in configs}
        assert len(shard_layout_keys) == 1, (
            f"rollout engines {assignment.rollout_engine_indices} hold rank {assignment.rollout_engine_rank} in "
            f"different layouts, so one model replica cannot serve them: {shard_layout_keys}"
        )
        configs_by_rollout_engine_rank[assignment.rollout_engine_rank] = configs[0]
    return configs_by_rollout_engine_rank


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    valid_fields = set(_record_field_names(ServerArgs))
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


def _query_config(rollout_engine: SGLangApiClient, rollout_engine_rank: int) -> RolloutEngineRankConfig:
    parallelism_info = async_utils.run(rollout_engine.get_parallelism_info(rank=rollout_engine_rank))
    server_info = async_utils.run(rollout_engine.get_server_info())
    return RolloutEngineRankConfig(
        parallelism=RankParallelismConfig.from_dict(parallelism_info),
        server_args=create_server_args_from_dict(server_info),
    )
