import socket
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, nullcontext

import ray
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState, get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.distributed_lock import create_world_ticket_lock
from miles.utils.distributed_utils import init_process_group


class UpdateWeightFromDistributed(WeightTransferProtocol):
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._model_update_groups = None
        parallel_state = get_parallel_state()
        self._engine_lock: AbstractContextManager = (
            create_world_ticket_lock(
                prefix="miles/weight_update",
                participates=parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0,
            )
            if parallel_state.pp.size > 1
            else nullcontext()
        )

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self._selector = selector
        self._engine_gpu_counts = engine_gpu_counts

        # One sender per replica set; one NCCL group (sender + all engines) per shard.
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        shard = 0 if placement.gather_pp else parallel_state.pp.rank
        if self.is_sender:
            self.group_name = f"miles-pp_{shard}"
            disconnect_rollout_engines_from_distributed(
                self.args, self.group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self.group_name, rollout_engines
            )

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        """Lock → broadcast → clear → unlock. Lock prevents NCCL deadlock."""
        with self._engine_lock:
            futures = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                self.rollout_engines,
                bucket,
                selector=self._selector,
            )
            async_utils.wait_futures(futures)
            bucket.clear()


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[SGLangApiClient],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

    futures = []
    rank_cursor = 1
    for i, api_client in enumerate(rollout_engines):
        futures.append(
            async_utils.submit(
                api_client.init_weights_update_group(
                    master_address,
                    master_port,
                    rank_cursor,
                    world_size,
                    group_name,
                    backend="nccl",
                )
            )
        )
        rank_cursor += engine_gpu_counts[i]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    async_utils.wait_futures(futures)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    futures = [async_utils.submit(client.destroy_weights_update_group(group_name)) for client in rollout_engines]
    try:
        if model_update_groups is not None:
            dist.destroy_process_group(model_update_groups)
    finally:
        async_utils.wait_futures(futures)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    rollout_engines: Sequence[SGLangApiClient],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    selector: str = "all",
) -> list[Future]:
    """
    Send metadata (HTTP), broadcast tensors (NCCL rank 0 → engines).
    """
    futures = [
        async_utils.submit(
            client.update_weights_from_distributed(
                names=[name for name, _ in converted_named_tensors],
                dtypes=[param.dtype for _, param in converted_named_tensors],
                shapes=[param.shape for _, param in converted_named_tensors],
                selector=selector,
                group_name=group_name,
            )
        )
        for client in rollout_engines
    ]

    contiguous_tensors = [
        param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in converted_named_tensors
    ]
    handles = []
    for tensor in contiguous_tensors:
        handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return futures
