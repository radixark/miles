import dataclasses
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import ray
import torch
from sglang.srt.server_args import ServerArgs
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.shared_utils.engine_rank_assignment import (
    EngineRankAssignment,
)
from miles.utils import async_utils
from miles.utils.workers.argv_utils import _record_field_names

logger = logging.getLogger(__name__)


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
    valid_fields = set(_record_field_names(ServerArgs))
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
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    local_ip = ray._private.services.get_node_ip_address()
    transfer_engine.initialize(local_ip, "P2PHANDSHAKE", "rdma", "")
    return transfer_engine


def query_remote_weight_infos(
    rollout_engines: Sequence[SGLangApiClient],
    assignments: Sequence[EngineRankAssignment],
) -> tuple[dict, dict, dict]:
    """Query remote rollout engines for weight info, session IDs, and server args."""
    remote_weight_infos_by_session_id = {}
    targets_to_session_id = {}
    session_id_to_server_args = {}
    targets_to_query = {
        (engine_index, assignment.engine_rank)
        for assignment in assignments
        for engine_index in assignment.engine_indices
    }

    for engine_ind, engine_rank in targets_to_query:
        session_id, weights_info = async_utils.run(
            rollout_engines[engine_ind].get_remote_instance_transfer_engine_info(rank=engine_rank)
        )
        parallelism_info = async_utils.run(rollout_engines[engine_ind].get_parallelism_info(rank=engine_rank))

        session_id_to_server_args[session_id] = create_server_args_from_dict(
            async_utils.run(rollout_engines[engine_ind].get_server_info())
        )
        assert session_id is not None, f"Failed to get session id from rollout engine {engine_ind} rank {engine_rank}"
        remote_weight_infos_by_session_id[session_id] = (weights_info, parallelism_info)
        targets_to_session_id[(engine_ind, engine_rank)] = session_id

    return remote_weight_infos_by_session_id, targets_to_session_id, session_id_to_server_args
