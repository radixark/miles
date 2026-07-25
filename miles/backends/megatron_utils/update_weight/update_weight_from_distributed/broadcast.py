import os
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils import accelerator
from miles.utils.distributed_utils import init_process_group
from miles.utils.http_utils import is_port_available

from miles.utils.lora import LORA_ADAPTER_NAME
from ..common import _check_weight_sync_results
from .mixin import DistBucketedWeightUpdateMixin


def _get_update_weight_group_index(group_name: str) -> int:
    prefix = "miles-pp_"
    if group_name.startswith(prefix):
        suffix = group_name[len(prefix) :]
        if suffix.isdigit():
            return int(suffix)
        if "-engine_" in suffix:
            pp_rank, engine_rank = suffix.split("-engine_", maxsplit=1)
            if pp_rank.isdigit() and engine_rank.isdigit():
                return int(pp_rank) * 32 + int(engine_rank)
    return sum(ord(char) for char in group_name) % 32


def _get_update_weight_master_port(args: Namespace, group_name: str, consecutive: int = 8) -> int:
    base = args.update_weight_master_port_base
    stride = args.update_weight_master_port_stride
    retries = args.update_weight_master_port_retries
    group_index = _get_update_weight_group_index(group_name)
    start_port = base + group_index * stride
    scan_window = min(retries, stride)
    if scan_window < consecutive:
        raise RuntimeError(
            f"Invalid update-weight port config for {group_name}: "
            f"base={base}, stride={stride}, retries={retries}, consecutive={consecutive}"
        )

    for master_port in range(start_port, start_port + scan_window - consecutive + 1):
        if all(is_port_available(master_port + offset) for offset in range(consecutive)):
            return master_port
    raise RuntimeError(
        f"Failed to find update-weight master port for {group_name}: "
        f"base={base}, stride={stride}, retries={retries}, start_port={start_port}, "
        f"scan_window={scan_window}, consecutive={consecutive}"
    )


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
        self._weight_update_connections: list[tuple[str, dist.ProcessGroup, Sequence[ActorHandle]]] = []
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale: bool = False
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
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        pp_rank = get_parallel_state().pp.rank
        if self._is_source:
            self._group_name = f"miles-pp_{pp_rank}"

        if self._is_source:
            for group_name, process_group, engines in self._weight_update_connections:
                disconnect_rollout_engines_from_distributed(self.args, group_name, process_group, engines)

            if engine_gpu_counts is None:
                engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
            if len(engine_gpu_counts) != len(rollout_engines):
                raise ValueError(
                    f"engine_gpu_counts has {len(engine_gpu_counts)} entries for {len(rollout_engines)} engines"
                )

            update_mode = os.environ.get("UPDATE_MODE", "broadcast")
            split_engines = update_mode in {"p2p-broadcast", "gloo-broadcast"} or (
                accelerator.is_musa_environment() and update_mode == "sync-broadcast"
            )
            if split_engines and len(rollout_engines) > 1 and not self.is_lora:
                self._weight_update_connections = []
                for engine_index, (engine, gpu_count) in enumerate(zip(rollout_engines, engine_gpu_counts)):
                    group_name = f"{self._group_name}-engine_{engine_index}"
                    process_group = connect_rollout_engines_from_distributed(
                        self.args,
                        group_name,
                        [engine],
                        engine_gpu_counts=[gpu_count],
                    )
                    self._weight_update_connections.append((group_name, process_group, [engine]))
            else:
                process_group = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    rollout_engines,
                    engine_gpu_counts=engine_gpu_counts,
                )
                self._weight_update_connections = [(self._group_name, process_group, rollout_engines)]

            groups = [connection[1] for connection in self._weight_update_connections]
            self._model_update_groups = groups[0] if len(groups) == 1 else groups

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
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        try:
            for group_name, process_group, engines in self._weight_update_connections:
                refs = update_weights_from_distributed(
                    group_name,
                    process_group,
                    self.weight_version,
                    engines,
                    converted_named_tensors,
                )
                ray.get(refs)
            converted_named_tensors.clear()
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
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

        contiguous_tensors = [
            param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in named_tensors
        ]
        for group_name, process_group, engines in self._weight_update_connections:
            refs = [
                engine.load_lora_adapter_from_distributed.remote(
                    lora_name=LORA_ADAPTER_NAME,
                    config_dict=self._lora_config,
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=group_name,
                )
                for engine in engines
            ]
            handles = [dist.broadcast(tensor, 0, group=process_group, async_op=True) for tensor in contiguous_tensors]
            for handle in handles:
                handle.wait()

            _check_weight_sync_results(ray.get(refs), is_lora=True)

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

        # NCCL needs contiguous buffers (lora_B slices are strided); the list keeps them alive
        # until the async broadcasts complete.
        broadcast_tensors = [param.data.contiguous() for _, param in named_tensors]
        for group_name, process_group, engines in self._weight_update_connections:
            refs = [
                engine.load_lora_adapter_from_distributed.remote(
                    lora_name=lora_name,
                    config_dict=lora_config,
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=group_name,
                    upsert=True,
                )
                for engine in engines
            ]
            handles = [dist.broadcast(tensor, 0, group=process_group, async_op=True) for tensor in broadcast_tensors]
            for handle in handles:
                handle.wait()

            _check_weight_sync_results(ray.get(refs), is_lora=True)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
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
    master_port = _get_update_weight_master_port(args, group_name)
    world_size = sum(engine_gpu_counts) + 1
    backend = accelerator.weight_update_backend()

    refs = []
    rank_cursor = 1
    for i, engine in enumerate(rollout_engines):
        refs.append(
            engine.init_weights_update_group.remote(
                master_address,
                master_port,
                rank_cursor,
                world_size,
                group_name,
                backend=backend,
            )
        )
        rank_cursor += engine_gpu_counts[i]
    model_update_groups = init_process_group(
        backend=backend,
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    try:
        if model_update_groups is not None:
            dist.destroy_process_group(model_update_groups)
    finally:
        ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata over Ray, then transfer tensors according to UPDATE_MODE.
    """
    update_mode = os.environ.get("UPDATE_MODE", "broadcast")
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]
    if update_mode == "p2p-broadcast":
        for i, (_, param) in enumerate(converted_named_tensors):
            dist.send(param.data, dst=1, group=group, tag=i)
    elif update_mode == "gloo-broadcast":
        for _, param in converted_named_tensors:
            dist.broadcast(param.data.cpu(), 0, group=group, async_op=False)
    elif update_mode == "sync-broadcast":
        for _, param in converted_named_tensors:
            dist.broadcast(param.data, 0, group=group, async_op=False)
    else:
        contiguous_tensors = [
            param.data if param.data.is_contiguous() else param.data.contiguous()
            for _, param in converted_named_tensors
        ]
        handles = []
        for tensor in contiguous_tensors:
            handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
        for handle in handles:
            handle.wait()

    return refs
