import inspect
import re
import socket
import time
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence
from typing import Callable

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from ray import ObjectRef
from ray.actor import ActorHandle

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except:
    from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group, init_process_group
from miles.utils.types import ParamInfo

from .megatron_to_hf import convert_to_hf  # noqa: F401

try:
    try:
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
    except ImportError:
        from sglang.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

    use_flattened_tensor_bucket = True
except Exception:
    use_flattened_tensor_bucket = False

def connect_rollout_engines_from_distributed(
        args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = len(rollout_engines) * args.rollout_num_gpus_per_engine + 1

    refs = [
        engine.init_weights_update_group.remote(
            master_address,
            master_port,
            i * args.rollout_num_gpus_per_engine + 1,
            world_size,
            group_name,
            backend="nccl",
            )
        for i, engine in enumerate(rollout_engines)
    ]
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
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_distributed(
        args: Namespace,
        group_name: str,
        group: dist.ProcessGroup,
        weight_version: int,
        rollout_engines: Sequence[ActorHandle],
        converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
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

    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
