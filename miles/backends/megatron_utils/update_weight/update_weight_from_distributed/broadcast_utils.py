import os
import socket
import time
from argparse import Namespace
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from ray.actor import ActorHandle

from miles.utils.distributed_utils import init_process_group


_DEFAULT_WEIGHT_SYNC_P2P_OPS_PER_BATCH = 64


def _run_batched_p2p_ops(ops):
    ops_per_batch = int(
        os.environ.get(
            "WEIGHT_SYNC_P2P_OPS_PER_BATCH",
            _DEFAULT_WEIGHT_SYNC_P2P_OPS_PER_BATCH,
        )
    )

    batch = []

    def flush_batch():
        if not batch:
            return
        for work in dist.batch_isend_irecv(batch):
            work.wait()
        batch.clear()

    for op in ops:
        batch.append(op)
        if len(batch) >= ops_per_batch:
            flush_batch()
    flush_batch()


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


def connect_rollout_relay_from_distributed(
    group_name: str,
    relay_engine: ActorHandle,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + relay engine TP0. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = 2

    ref = relay_engine.init_weights_update_group.remote(
        master_address,
        master_port,
        1,
        world_size,
        group_name,
        backend="nccl",
        transfer_mode="relay",
    )
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get([ref])
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def acquire_rollout_engine_lock(rollout_engine_lock: ActorHandle) -> None:
    while not ray.get(rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int | None,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 -> engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version) if weight_version is not None else None,
        )
        for engine in rollout_engines
    ]

    handles = [
        dist.broadcast(
            param.data,
            0,
            group=group,
            async_op=True,
        )
        for _, param in converted_named_tensors
    ]
    for handle in handles:
        handle.wait()

    return refs


def update_weights_from_distributed_send_recv(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int | None,
    rollout_engine: ActorHandle,
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> ObjectRef:
    """
    Send metadata (Ray) to the relay engine, then send tensors with NCCL
    send/recv from trainer rank 0 to the relay TP0 (peer=1 in the 2-rank group).
    """
    group_world_size = dist.get_world_size(group)
    if group_world_size != 2:
        raise ValueError(
            "sendrecv broadcast expects a trainer-to-relay process group with "
            f"world_size=2, but got {group_world_size}."
        )

    ref = rollout_engine.update_weights_from_distributed.remote(
        names=[name for name, _ in converted_named_tensors],
        dtypes=[param.dtype for _, param in converted_named_tensors],
        shapes=[param.shape for _, param in converted_named_tensors],
        group_name=group_name,
        weight_version=str(weight_version) if weight_version is not None else None,
        transfer_mode="relay",
    )

    _run_batched_p2p_ops(
        [
            dist.P2POp(
                dist.isend,
                param,
                group=group,
                group_peer=1,
            )
            for _, param in converted_named_tensors
        ]
    )

    return ref
