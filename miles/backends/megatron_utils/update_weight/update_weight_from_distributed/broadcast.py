import socket
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils import async_utils
from miles.utils.distributed_lock import create_world_ticket_lock
from miles.utils.distributed_utils import init_process_group
from miles.utils.lora import LORA_ADAPTER_NAME

from ..common import _check_weight_sync_results
from .mixin import DistBucketedWeightUpdateMixin

if TYPE_CHECKING:
    pass


class UpdateWeightFromDistributed(DistBucketedWeightUpdateMixin):
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self.rollout_engines: Sequence[SGLangApiClient] | None = None
        self._connection_stale: bool = False
        self._engine_lock: AbstractContextManager = (
            create_world_ticket_lock(prefix="miles/weight_update", participates=self._is_source)
            if get_parallel_state().pp.size > 1
            else nullcontext()
        )
        self._init_lora(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
            is_lora=is_lora,
        )

    # TODO: avoid dup code during yueming's refactor (temp write this to avoid introducing potentially conflicting base class)
    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        pp_rank = get_parallel_state().pp.rank
        if self._is_source:
            self._group_name = f"miles-pp_{pp_rank}"

        if self._is_source:
            disconnect_rollout_engines_from_distributed(
                self.args, self._group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, rollout_engines
            )

    @property
    def _is_source(self):
        """If it's the source gpu that broadcasting weights to rollout side"""
        return get_parallel_state().intra_dp_cp.rank == 0 and get_parallel_state().tp.rank == 0

    @property
    def _is_lora_source(self) -> bool:
        """The single rank holding the full adapter (DP=TP=PP=0). At PP=1 (enforced
        for LoRA) this coincides with ``_is_source``."""
        ps = get_parallel_state()
        return ps.pp.rank == 0 and ps.tp.rank == 0 and ps.intra_dp_cp.rank == 0

    def _update_weight_implementation(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        """Lock → broadcast → clear → unlock. Lock prevents NCCL deadlock."""
        with self._engine_lock:
            futures = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.rollout_engines,
                converted_named_tensors,
                selector=self._weight_update_selector,
            )
            async_utils.wait_futures(futures)
            converted_named_tensors.clear()
        if pbar:
            pbar.update(1)

    def _update_lora_weight_implementation(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Send adapter metadata over Ray, then broadcast the tensors (src=0).

        Reuses the base broadcast group (``self._model_update_groups`` /
        ``self._group_name``); base and adapter syncs are strictly sequential, so
        sharing the NCCL communicator is safe. No CUDA IPC, so it works across
        nodes: the engine allocates buffers from the metadata and broadcast-receives
        in order.
        """
        names = [name for name, _ in named_tensors]
        dtypes = [param.dtype for _, param in named_tensors]
        shapes = [list(param.shape) for _, param in named_tensors]

        futures = [
            async_utils.submit(
                client.load_lora_adapter_from_distributed(
                    lora_name=LORA_ADAPTER_NAME,
                    config_dict=self._lora_config,
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=self._group_name,
                )
            )
            for client in self.rollout_engines
        ]
        contiguous_tensors = [
            param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in named_tensors
        ]
        handles = [
            dist.broadcast(tensor, 0, group=self._model_update_groups, async_op=True) for tensor in contiguous_tensors
        ]
        for handle in handles:
            handle.wait()

        _check_weight_sync_results(async_utils.wait_futures(futures), is_lora=True)

    def _update_multi_lora_weight_implementation(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
        *,
        lora_name: str,
        lora_config: dict,
    ) -> None:
        """Multi-LoRA variant of ``_update_lora_weight_implementation``: same transport, but with a
        per-adapter slot name/config and an upsert RPC (in-place insert-or-overwrite, no unload/register)."""
        names = [name for name, _ in named_tensors]
        dtypes = [param.dtype for _, param in named_tensors]
        shapes = [list(param.shape) for _, param in named_tensors]

        futures = [
            async_utils.submit(
                client.load_lora_adapter_from_distributed(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=self._group_name,
                    upsert=True,
                )
            )
            for client in self.rollout_engines
        ]
        # NCCL needs contiguous buffers (lora_B slices are strided); the list keeps them alive
        # until the async broadcasts complete.
        broadcast_tensors = [param.data.contiguous() for _, param in named_tensors]
        handles = [
            dist.broadcast(tensor, 0, group=self._model_update_groups, async_op=True) for tensor in broadcast_tensors
        ]
        for handle in handles:
            handle.wait()

        _check_weight_sync_results(async_utils.wait_futures(futures), is_lora=True)


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
    weight_version: int,
    rollout_engines: Sequence[SGLangApiClient],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    selector: str = "all",
) -> list[Future]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    futures = [
        async_utils.submit(
            client.update_weights_from_distributed(
                names=[name for name, _ in converted_named_tensors],
                dtypes=[param.dtype for _, param in converted_named_tensors],
                shapes=[param.shape for _, param in converted_named_tensors],
                selector=selector,
                group_name=group_name,
                weight_version=str(weight_version),
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
