import socket
import time
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.session import check_weight_sync_results, weight_update_selector
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
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
        self._selector = weight_update_selector(args)

    def connect(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self._connection_stale = False
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # One sender per replica set; one NCCL group (sender + all engines) per shard.
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        shard = 0 if placement.gather_pp else parallel_state.pp.rank
        self.is_lora_sender = self.is_sender and shard == 0
        if self.is_sender:
            self.group_name = f"miles-pp_{shard}"
            disconnect_rollout_engines_from_distributed(
                self.args, self.group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self.group_name, rollout_engines
            )

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]], weight_version: int) -> None:
        """Serialize NCCL broadcasts and always release the rollout lock."""
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        try:
            refs = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                weight_version,
                self.rollout_engines,
                bucket,
                selector=self._selector,
            )
            ray.get(refs)
            bucket.clear()
        finally:
            # Leaking this lock makes the next weight sync poll forever, so the
            # release must run after both successful and failed broadcasts.
            ray.get(self.rollout_engine_lock.release.remote())

    def send_adapter(
        self, named_tensors: list[tuple[str, torch.Tensor]], *, lora_name: str, lora_config: dict, upsert: bool
    ) -> None:
        """Send adapter metadata over Ray, then broadcast the tensors (src=0).

        Reuses the base broadcast group (``self._model_update_groups`` /
        ``self.group_name``); base and adapter syncs are strictly sequential, so
        sharing the NCCL communicator is safe. No CUDA IPC, so it works across
        nodes: the engine allocates buffers from the metadata and broadcast-receives
        in order. ``upsert`` maps to the engine's in-place insert-or-overwrite RPC
        (multi-LoRA slots); without it the caller unloads the old adapter first.
        """
        names = [name for name, _ in named_tensors]
        dtypes = [param.dtype for _, param in named_tensors]
        shapes = [list(param.shape) for _, param in named_tensors]

        refs = [
            engine.load_lora_adapter_from_distributed.remote(
                lora_name=lora_name,
                config_dict=lora_config,
                names=names,
                dtypes=dtypes,
                shapes=shapes,
                group_name=self.group_name,
                **({"upsert": True} if upsert else {}),
            )
            for engine in self.rollout_engines
        ]
        # NCCL needs contiguous buffers (lora_B slices are strided); the list keeps them
        # alive until the async broadcasts complete.
        contiguous_tensors = [
            param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in named_tensors
        ]
        handles = [
            dist.broadcast(tensor, 0, group=self._model_update_groups, async_op=True) for tensor in contiguous_tensors
        ]
        for handle in handles:
            handle.wait()

        check_weight_sync_results(ray.get(refs), is_lora=True)


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
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

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
                backend="nccl",
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
    selector: str = "all",
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            selector=selector,
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    contiguous_tensors = [
        param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in converted_named_tensors
    ]
    handles = []
    for tensor in contiguous_tensors:
        handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
