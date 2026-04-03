import dataclasses
import logging
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from mooncake.engine import TransferEngine
from ray.actor import ActorHandle
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TransferTaskP2PMeta:
    """Specifies a engine rollout rank to connect to."""

    # The index of the target rollout engine.
    engine_ind: int
    # The rank of the target shard within the rollout engine (corresponds to `sglang_tp_rank`).
    engine_rank: int
    # The source pp shard index.
    source_shard: int = 0


class RemoteTransferPlan:
    """
    Plans and manages remote weight transfers for p2p backends,
    assuming static training and rollout placements.
    """

    def __init__(self, args: Namespace, model: Sequence[torch.nn.Module]) -> None:
        self._get_parallelism(args)

    def _get_parallelism(self, args: Namespace) -> None:
        """
        Collect parallelism information for source (trainer) and target (rollout engines).

        All ranks with the same PP rank (gathered_dp_rank) hold a complete weight replica
        after bucketed all-gather across TP/EP/ETP dims. The size of this group is
        gathered_dp_size, and each rank has a unique gathered_dp_rank in [0, gathered_dp_size).
        """
        self._pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._pp_size = mpu.get_pipeline_model_parallel_world_size()

        world_size = dist.get_world_size()
        self._gathered_dp_size = world_size // self._pp_size

        my_pp_group = dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group())
        my_column_id = min(my_pp_group)
        all_column_ids = [None] * world_size
        dist.all_gather_object(all_column_ids, my_column_id)
        sorted_columns = sorted(set(all_column_ids))
        self._gathered_dp_rank = sorted_columns.index(my_column_id)

        self._rollout_pp_size = args.sglang_pp_size
        if self._rollout_pp_size != 1:
            raise NotImplementedError("Rollout pipeline parallelism is not tested yet.")
        self._rollout_num_gpu_per_engine = args.rollout_num_gpus_per_engine
        self._rollout_engine_count = args.rollout_num_gpus // self._rollout_num_gpu_per_engine
        self._rollout_num_gpus = args.rollout_num_gpus

    def plan_p2p(self) -> list[TransferTaskP2PMeta]:
        """
        Plan P2P transfer within each pp_group -> all target engine ranks.

        For each pp shard source rank, it plans the mapping relationship between n source dp ranks, m target rollout engines with k ranks each.
        The Transfer Plan Mapping Heuristics works as follows:
        1. for each target engine (idx, rank), assign source ranks in a round-robin manner until all source ranks are assigned at least once.
        2. for the reminder target (idx, rank), assign them to source ranks by priotizing the source with existing assignmeng of same rank.

        For example, 4 source ranks (0,1,2,3), 2 target engines with 3 ranks each (0,0),(0,1),(0,2),(1,0),(1,1),(1,2).
        The first round of assignment:
        source_rank=0 -> target (0,0)
        source_rank=1 -> target (0,1)
        source_rank=2 -> target (0,2)
        source_rank=3 -> target (1,0)
        The reminder assignment:
        source_rank=1 -> target (1,1)  # prioritize source_rank=1 as it had (0,1) assigned already.
        source_rank=2 -> target (1,2)

        Finally extract the transfer tasks matching the current dp_rank(self._gathered_dp_rank).

        """
        all_targets = [
            (engine_idx, engine_rank)
            for engine_idx in range(self._rollout_engine_count)
            for engine_rank in range(self._rollout_num_gpu_per_engine)
        ]
        assignments = defaultdict(lambda: defaultdict(list))

        # Total number of source-to-target P2P connections established.
        p2p_count = 0
        # step 1: assign engine ranks in a round-robin way
        for source_rank, (_, target) in zip(range(self._gathered_dp_size), enumerate(all_targets), strict=False):
            p2p_count += 1
            engine_idx, engine_rank = target
            assignments[source_rank][engine_rank].append(engine_idx)

        def count_engine_index_assignments(engine_rank: int) -> list[int]:
            return [len(assignments[source][engine_rank]) for source in range(self._gathered_dp_size)]

        cur_source_index = 0
        # step 2: assign the left engine ranks.
        if p2p_count < len(all_targets):
            for target in all_targets[p2p_count:]:
                engine_idx, engine_rank = target
                counted = count_engine_index_assignments(engine_rank)
                if max(counted) > 0:
                    # assign it to existing source rank assigned to the same target engine_rank, with lowest load
                    _, select_source = min((val, idx) for (idx, val) in enumerate(counted) if val > 0)
                else:
                    # otherwise round robin
                    select_source = cur_source_index % self._gathered_dp_size
                    cur_source_index += 1
                assignments[select_source][engine_rank].append(engine_idx)

        transfer_tasks = []
        for engine_rank, engine_indices in assignments[self._gathered_dp_rank].items():
            for engine_ind in engine_indices:
                transfer_tasks.append(
                    TransferTaskP2PMeta(source_shard=self._pp_rank, engine_ind=engine_ind, engine_rank=engine_rank)
                )

        return transfer_tasks


@dataclasses.dataclass
class RemoteWeightInfo:
    """
    The remote weight info related to one specific engine_rank.
    """

    session_id: str
    weights_info: dict[str, tuple[int, int, int]]  # name -> (remote_address, numel, element_size)


class P2PTransferManager:
    """Generic async task manager for P2P writes.

    Accepts arbitrary callables via submit(), runs them in a thread pool,
    and tracks futures for bulk waiting.
    """

    def __init__(self, num_workers: int = 8, transfer_timeout: float = 30.0):
        self.num_workers = num_workers
        self.transfer_timeout = transfer_timeout
        self.executor: ThreadPoolExecutor | None = None
        self.transfer_futures: list[Future] = []

    def ensure_started(self) -> None:
        if self.executor is None:
            # NOTE: RDMA ops won't be affected by the python GIL
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def submit(self, fn: Callable, *args) -> None:
        """Submit a callable to the thread pool."""
        self.ensure_started()
        future = self.executor.submit(fn, *args)
        self.transfer_futures.append(future)

    def submit_returning_future(self, fn: Callable, *args) -> torch.Future:
        """Submit a callable and return its future (also tracked for bulk waiting)."""
        self.ensure_started()
        future = self.executor.submit(fn, *args)
        self.transfer_futures.append(future)
        return future

    def wait_transfers(self) -> None:
        """Wait for all submitted tasks to complete."""
        for future in self.transfer_futures:
            try:
                future.result(timeout=self.transfer_timeout)
            except Exception as e:
                logger.error(f"[P2P] Transfer future failed: {e}")

        self.transfer_futures.clear()


def create_server_args_from_dict(data_dict: dict) -> ServerArgs:
    valid_fields = {f.name for f in dataclasses.fields(ServerArgs)}
    filtered_data = {k: v for k, v in data_dict.items() if k in valid_fields}
    return ServerArgs(**filtered_data)


def register_cpu_memory(params_dict: dict, transfer_engine) -> dict:
    """Register CPU pinned memory with the transfer engine."""
    weight_dict = {}

    for name, cpu_tensor in params_dict.items():
        addr = cpu_tensor.data_ptr()
        size = cpu_tensor.numel() * cpu_tensor.element_size()
        # NOTE: theoretically using huge page allocator
        # in torch backend could imporve registration speed.
        ret = transfer_engine.register_memory(addr, size)
        if ret != 0:
            raise RuntimeError(f"register CPU memory failed for weight {name}, error: {ret}")
        weight_dict[name] = (addr, cpu_tensor.numel(), cpu_tensor.element_size())

    return weight_dict


def create_transfer_engine():
    transfer_engine = TransferEngine()
    local_ip = ray._private.services.get_node_ip_address()
    transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")
    return transfer_engine


def query_remote_weight_infos(
    rollout_engines: Sequence[ActorHandle],
    targets,
) -> tuple[dict, dict, dict]:
    """Query remote rollout engines for weight info, session IDs, and server args."""
    remote_weight_infos_by_session_id = {}
    targets_to_session_id = {}
    session_id_to_server_args = {}
    targets_to_query = set((target.engine_ind, target.engine_rank) for target in targets)

    for engine_ind, engine_rank in targets_to_query:
        session_id, weights_info = ray.get(
            rollout_engines[engine_ind].get_remote_instance_transfer_engine_info.remote(rank=engine_rank)
        )
        parallelism_info = ray.get(rollout_engines[engine_ind].get_parallelism_info.remote(rank=engine_rank))

        session_id_to_server_args[session_id] = create_server_args_from_dict(
            ray.get(rollout_engines[engine_ind].get_server_info.remote())
        )
        assert session_id is not None, f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
        remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
        targets_to_session_id[(engine_ind, engine_rank)] = session_id

    return remote_weight_infos_by_session_id, targets_to_session_id, session_id_to_server_args
