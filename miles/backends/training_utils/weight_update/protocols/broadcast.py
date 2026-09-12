import re
import socket
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, nullcontext

import ray
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.utils import async_utils
from miles.utils.distributed_lock import create_world_ticket_lock
from miles.utils.distributed_utils import get_gloo_group, init_process_group

_GLOBAL_EXPERT_HF_NAME = re.compile(r"(?:^|\.)experts\.\d+\.")


class UpdateWeightFromDistributed(WeightTransferProtocol):
    """
    Update distributed engines via NCCL from one sender per retained PP/EP shard.
    One sender per PP shard also owns its dense, router and shared-expert weights.
    """

    supports_lora = True
    required_placement = WeightUpdatePlacement(gather_pp=False, gather_ep=False)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._model_update_groups = None
        self._engine_lock: AbstractContextManager = nullcontext()
        self._placement: WeightUpdatePlacement | None = None
        self._sends_dense = False

    def configure(self, parallel_state: ParallelState, placement: WeightUpdatePlacement) -> None:
        """Select one sender per retained PP/EP shard and build their shared lock."""
        assert self._placement is None, "broadcast protocol topology is already configured"
        assert placement.gather_tp, "distributed broadcast requires gathered TP/ETP weights"

        pp_shard = 0 if placement.gather_pp else parallel_state.pp.rank
        ep_shard = 0 if placement.gather_ep else parallel_state.ep.rank
        group = get_gloo_group()
        local_coordinate = (dist.get_rank(), pp_shard, ep_shard)
        all_coordinates: list = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(all_coordinates, local_coordinate, group=group)

        sender_by_shard: dict[tuple[int, int], int] = {}
        for rank, rank_pp_shard, rank_ep_shard in all_coordinates:
            shard = (rank_pp_shard, rank_ep_shard)
            sender_by_shard[shard] = min(rank, sender_by_shard.get(shard, rank))

        rank = dist.get_rank()
        sender_rank = sender_by_shard[(pp_shard, ep_shard)]
        dense_sender_rank = min(
            candidate_rank
            for (candidate_pp_shard, _candidate_ep_shard), candidate_rank in sender_by_shard.items()
            if candidate_pp_shard == pp_shard
        )
        self.is_sender = rank == sender_rank
        self._sends_dense = rank == dense_sender_rank
        self.group_name = f"miles-pp_{pp_shard}" if placement.gather_ep else f"miles-pp_{pp_shard}-ep_{ep_shard}"
        if len(sender_by_shard) > 1:
            self._engine_lock = create_world_ticket_lock(
                prefix="miles/weight_update",
                participates=self.is_sender,
            )
        self._placement = placement

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """Create this sender's NCCL group under the shared engine lock."""
        assert self._placement == placement, "connect placement differs from configured broadcast topology"
        assert self.is_sender is not None, "configure() must set is_sender before connect()"
        self.rollout_engines = rollout_engines
        self._selector = selector
        self._engine_gpu_counts = engine_gpu_counts

        if self.is_sender:
            with self._engine_lock:
                disconnect_rollout_engines_from_distributed(
                    self.args, self.group_name, self._model_update_groups, self.rollout_engines
                )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self.group_name,
                    rollout_engines,
                    engine_gpu_counts=engine_gpu_counts,
                )

    def should_send_weight_unit(self, unit: list[tuple[str, torch.Tensor]]) -> bool:
        """Dense owner sends its full PP slice; other EP owners send routed experts only."""
        if self._sends_dense:
            return True
        routed = [_GLOBAL_EXPERT_HF_NAME.search(name) is not None for name, _tensor in unit]
        assert routed and (
            all(routed) or not any(routed)
        ), f"Weight update unit mixes routed-expert and dense tensors: {[name for name, _tensor in unit]}"
        return all(routed)

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
