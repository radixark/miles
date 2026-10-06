from collections.abc import Sequence

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.utils import async_utils
from miles.utils.workers.argv_utils import _record_field_names


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    valid_fields = set(_record_field_names(ServerArgs))
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


def query_remote_weight_infos(
    rollout_engines: Sequence[SGLangApiClient],
    assignments: Sequence[RolloutEngineRankAssignment],
) -> tuple[dict, dict, dict]:
    """Query remote rollout engines for weight info, session IDs, and server args."""
    remote_weight_infos_by_session_id = {}
    targets_to_session_id = {}
    session_id_to_server_args = {}
    targets_to_query = {
        (rollout_engine_ind, assignment.rollout_engine_rank)
        for assignment in assignments
        for rollout_engine_ind in assignment.rollout_engine_indices
    }

    for rollout_engine_ind, rollout_engine_rank in targets_to_query:
        session_id, weights_info = async_utils.run(
            rollout_engines[rollout_engine_ind].get_remote_instance_transfer_engine_info(rank=rollout_engine_rank)
        )
        parallelism_info = async_utils.run(
            rollout_engines[rollout_engine_ind].get_parallelism_info(rank=rollout_engine_rank)
        )

        session_id_to_server_args[session_id] = create_server_args_from_dict(
            async_utils.run(rollout_engines[rollout_engine_ind].get_server_info())
        )
        assert (
            session_id is not None
        ), f"Failed to get session id from rollout engine {rollout_engine_ind} rank {rollout_engine_rank}"
        remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
        targets_to_session_id[(rollout_engine_ind, rollout_engine_rank)] = session_id

    return remote_weight_infos_by_session_id, targets_to_session_id, session_id_to_server_args
