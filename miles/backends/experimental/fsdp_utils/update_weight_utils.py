import abc
import logging
import socket
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from torch.distributed.tensor import DTensor

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]
except ImportError:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions  # type: ignore[import]

from sglang.srt.utils import MultiprocessingSerializer

from miles.utils.distributed_utils import get_gloo_group, init_process_group

try:
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]


logger = logging.getLogger(__name__)


from .adaptations.weight_bridge import get_param_transform
from .dtensor import gather_full_param


def _iter_sync_named_params(name, param, model_type, model, orig_dtypes=None):
    """Yield (name, tensor) pairs for the rollout engine, applying the registered WeightBridge transform
    for this model_type (e.g. unfusing batched MoE experts); params with no transform stream unchanged.
    ``model`` is the HF module the params came from (transforms resolve its checkpoint-conversion mapping);
    ``orig_dtypes`` downcasts an fp32 master tensor back to its on-disk dtype."""
    expand = get_param_transform(name, param, model_type)
    if expand is None:
        yield name, param
        return

    # Materialize the full (unsharded) tensor before the transform slices it.
    full = gather_full_param(param)
    if orig_dtypes is not None:
        target = orig_dtypes.get(name)
        if target is not None and full.dtype != target:
            full = full.to(target)
    yield from expand(name, full, model)


class UpdateWeight(abc.ABC):
    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        self.args = args
        self.model = model
        self.weight_version = 0

    @abc.abstractmethod
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        pass

    def update_weights(self) -> None:
        self.weight_version += 1

        if dist.get_rank() == 0:
            futures = [engine.pause_generation.remote() for engine in self.rollout_engines]
            futures.extend([engine.flush_cache.remote() for engine in self.rollout_engines])
            ray.get(futures)
            ray.get([engine.begin_weight_update.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        bucket = []
        bucket_size = 0
        model_type = getattr(getattr(self.model, "config", None), "model_type", "")
        # FP32 masters are streamed in each parameter's original on-disk dtype.
        orig_dtypes = getattr(self.model, "_fsdp_sync_orig_dtypes", None)
        for raw_name, raw_param in self.model.state_dict().items():
            for name, param in _iter_sync_named_params(raw_name, raw_param, model_type, self.model, orig_dtypes):
                param_size = param.numel() * param.element_size()
                if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                    self.wait_and_update_bucket_weights(bucket)
                    del bucket
                    bucket = []
                    bucket_size = 0

                # passthrough params only (split experts are pre-cast); the target_dtype cast is deferred to post-wait
                target_dtype = orig_dtypes.get(name) if orig_dtypes is not None else None
                param = gather_full_param(param, async_op=True)
                bucket.append((name, param, target_dtype))
                bucket_size += param_size

        if bucket:
            self.wait_and_update_bucket_weights(bucket)
            del bucket
            bucket = []
            bucket_size = 0

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.end_weight_update.remote() for engine in self.rollout_engines])
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def wait_and_update_bucket_weights(self, bucket):
        resolved = []
        for name, param, target_dtype in bucket:
            if hasattr(param, "wait"):
                param = param.wait()
            # downcast the fp32 master to its on-disk dtype (round-to-nearest-even reproduces disk bf16); None = no cast
            if target_dtype is not None and param.dtype != target_dtype:
                param = param.to(target_dtype)
            resolved.append((name, param))
        self.update_bucket_weights(resolved, weight_version=self.weight_version)

    @abc.abstractmethod
    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        pass


class UpdateWeightFromTensor(UpdateWeight):
    """Push model weights to rollout engines as tensors, streamed in size-bounded buckets (optionally
    grouped + flattened per dtype, one RPC per dtype per bucket)."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """Attach rollout engines and create per-engine IPC (Gloo) groups (sets gather src rank, engine, tp_rank)."""
        self.rollout_engines = rollout_engines

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine
                self.tp_rank = dist.get_rank() - start_rank

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        monkey_patch_torch_reductions()
        logger.info("Using flattened tensor bucket")
        named_tensors_by_dtypes = {}
        for name, tensor in named_tensors:
            dtype = tensor.dtype
            if dtype not in named_tensors_by_dtypes:
                named_tensors_by_dtypes[dtype] = []
            named_tensors_by_dtypes[dtype].append((name, tensor))

        serialized_tensors = []
        for _dtype, named_tensors in named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
                "metadata": metadata,
            }
            serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

        if self._ipc_gather_src == dist.get_rank():
            # On rank 0, prepare a list to hold the gathered batches from all ranks.
            gathered_serialized_batches = [None for _ in range(dist.get_world_size(self._ipc_gather_group))]
        else:
            gathered_serialized_batches = None

        # Gather the serialized batches from all ranks to rank 0.
        dist.gather_object(
            obj=serialized_tensors,
            object_gather_list=gathered_serialized_batches,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        if dist.get_rank() == self._ipc_gather_src:
            # TODO: assumes all ranks have the same number of dtype buckets
            num_dtypes = len(gathered_serialized_batches[0])
            assert num_dtypes > 0
            for i in range(num_dtypes):
                kwargs = {
                    "serialized_named_tensors": [tensors[i] for tensors in gathered_serialized_batches],
                    "load_format": "flattened_bucket",
                    "flush_cache": False,
                    "weight_version": str(weight_version),
                }
                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                result = ray.get(ref)
                if isinstance(result, dict):
                    success = result.get("success", True)
                    error_msg = result.get("error_message") or result.get("message", "unknown error")
                else:
                    success = getattr(result, "success", True)
                    error_msg = getattr(result, "error_message", "unknown error")
                if not success:
                    raise RuntimeError(
                        f"Weight sync failed on rollout engine: {error_msg}. " f"Check SGLang version compatibility."
                    )

        if dist.get_rank() == self._ipc_gather_src:
            ref = self._ipc_engine.flush_cache.remote()
            ray.get(ref)


class UpdateWeightFromDistributed(UpdateWeight):
    """Broadcast weights via a temporary NCCL group to rollout engines."""

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """On rank 0, initialize a temporary NCCL group for parameter broadcast."""
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # TP weight sync: AllGather params to rank 0, then broadcast from rank 0 to all sglang engines
        self._is_src_rank = dist.get_rank() == 0
        if self._is_src_rank:
            self._group_name = "miles"
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            # +1 for the trainer's source rank (rank 0); rollout engine ranks start at 1
            world_size = self.args.rollout_num_gpus + 1

            refs = [
                engine.init_weights_update_group.remote(
                    master_address,
                    master_port,
                    i * self.args.rollout_num_gpus_per_engine + 1,
                    world_size,
                    self._group_name,
                    backend="nccl",
                )
                for i, engine in enumerate(self.rollout_engines)
            ]
            self._model_update_groups = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name=self._group_name,
            )
            ray.get(refs)

    def update_bucket_weights(self, named_tensors, weight_version=None) -> None:
        """Send names/dtypes/shapes metadata to engines, then broadcast the tensors (contiguous;
        DTensors materialized when world_size == 1)."""
        if not self._is_src_rank or not named_tensors:
            return

        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named_tensors],
                dtypes=[param.dtype for _, param in named_tensors],
                shapes=[param.shape for _, param in named_tensors],
                group_name=self._group_name,
                weight_version=str(weight_version),
            )
            for engine in self.rollout_engines
        ]

        handles = []
        # free cached blocks once before the batch (per-param empty_cache was a costly CUDA sync and ineffective)
        torch.cuda.empty_cache()
        for _name, param in named_tensors:
            param_data = param.data.contiguous()

            # avoid `DTensor._op_dispatcher.dispatch` has `assert compute_mesh is not None` error
            if dist.get_world_size() == 1 and isinstance(param_data, DTensor):
                param_data = param_data.full_tensor()

            handles.append(dist.broadcast(param_data, 0, group=self._model_update_groups, async_op=True))

        for handle in handles:
            handle.wait()
        ray.get(refs)
