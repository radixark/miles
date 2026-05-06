import gc
import logging
import os
import pickle
from collections.abc import Sequence
from datetime import timedelta
from typing import cast

import psutil
import torch

try:
    from torchft.checkpointing.pg_transport import (
        PGTransport,
        _StateDictMeta,
        _TensorMeta,
        _DTensorMeta,
        _cast_tensor,
    )
    from torch.distributed.tensor import DTensor
    from torch.utils._pytree import tree_flatten_with_path, tree_unflatten
except ImportError:
    PGTransport = None

from megatron.core.dist_checkpointing.tensor_aware_state_dict import MCoreTensorAwareStateDict

from miles.backends.megatron_utils.in_memory_checkpoint import InMemoryCheckpointManager, save_to_memory
from miles.utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)

# Must accommodate receiver's model init time (can take minutes for large models)
_DEFAULT_TIMEOUT = timedelta(seconds=600)


def _log_mem(tag: str) -> None:
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    cuda_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    cuda_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    logger.info(
        "[OOM_DEBUG][%s] pid=%d RSS=%.2fGB CUDA_alloc=%.2fGB CUDA_reserved=%.2fGB",
        tag, os.getpid(), rss, cuda_alloc, cuda_reserved,
    )


def send_ckpt(
    *,
    indep_dp: GroupInfo,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
    iteration: int,
    dst_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> None:
    """Send in-memory checkpoint to a destination cell via torchft PGTransport.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        model: Megatron model chunks.
        optimizer: Megatron optimizer.
        opt_param_scheduler: LR scheduler.
        iteration: Current training iteration / rollout_id.
        dst_rank: Destination alive_rank in the indep_dp process group.
        timeout: Timeout for the NCCL send operation.
    """
    _log_mem("send.enter")
    state_dict = save_to_memory(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )
    _log_mem("send.after_save_to_memory")

    payload = _serialize_for_transport(state_dict=state_dict, iteration=iteration)
    _log_mem("send.after_serialize")

    transport = _create_transport(indep_dp, timeout)
    transport.send_checkpoint(
        dst_ranks=[dst_rank],
        step=0,
        state_dict=payload,
        timeout=timeout,
    )
    _log_mem("send.after_send_checkpoint")
    transport.disallow_checkpoint()
    logger.info(f"Sent checkpoint (iteration={iteration}) to alive_rank={dst_rank}")


def recv_ckpt(
    *,
    indep_dp: GroupInfo,
    src_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> InMemoryCheckpointManager:
    """Receive checkpoint from a healthy cell via torchft PGTransport.

    Returns an InMemoryCheckpointManager containing the received state_dict,
    ready to be passed to initialize_model_and_optimizer.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        src_rank: Source alive_rank in the indep_dp process group.
        timeout: Timeout for the NCCL recv operation.

    Returns:
        InMemoryCheckpointManager with state_dict loaded, ready for
        initialize_model_and_optimizer to consume.
    """
    _log_mem("recv.enter")
    transport = _create_transport(indep_dp, timeout)
    payload = _recv_checkpoint_with_logging(
        transport=transport,
        src_rank=src_rank,
        step=0,
        timeout=timeout,
    )
    _log_mem("recv.after_recv_checkpoint")

    iteration, state_dict = _deserialize_from_transport(payload)
    _log_mem("recv.after_deserialize")
    logger.info(f"Received checkpoint (iteration={iteration}) from alive_rank={src_rank}")

    manager = InMemoryCheckpointManager()
    manager.save(state_dict, iteration=iteration)
    return manager


def _serialize_for_transport(
    *,
    state_dict: MCoreTensorAwareStateDict,
    iteration: int,
) -> dict[str, object]:
    tensors: list[torch.Tensor] = state_dict.pop_tensors()
    # torchft PGTransport._cast_tensor uses `type(t) is torch.Tensor` (strict),
    # which rejects torch.nn.Parameter. Detach into plain Tensors that share
    # storage but pass the type check.
    tensors = [t.detach() if type(t) is not torch.Tensor else t for t in tensors]
    total_bytes = sum(t.nbytes for t in tensors)
    sizes_gb = sorted([(t.nbytes / 1024**3, tuple(t.shape), str(t.dtype), str(t.device)) for t in tensors], reverse=True)
    top5 = sizes_gb[:5]
    logger.info(
        "[OOM_DEBUG] _serialize_for_transport: %d tensors, total %.2f GB; top5=%s",
        len(tensors),
        total_bytes / 1024**3,
        top5,
    )
    return {
        "tensors": tensors,
        "hollow_state_dict": state_dict,
        "iteration": iteration,
    }


def _deserialize_from_transport(
    payload: dict[str, object],
) -> tuple[int, MCoreTensorAwareStateDict]:
    iteration: int = payload["iteration"]
    hollow_state_dict: MCoreTensorAwareStateDict = payload["hollow_state_dict"]
    tensors: list[torch.Tensor] = payload["tensors"]

    total_bytes = sum(t.nbytes for t in tensors)
    logger.info(
        "[OOM_DEBUG] _deserialize_from_transport: %d tensors, total %.2f GB (devices=%s)",
        len(tensors),
        total_bytes / 1024**3,
        list({str(t.device) for t in tensors})[:5],
    )

    hollow_state_dict.insert_tensors(tensors)
    return iteration, hollow_state_dict


def _create_transport(indep_dp: GroupInfo, timeout: timedelta) -> PGTransport:
    return PGTransport(
        pg=indep_dp.group,
        timeout=timeout,
        device=torch.device("cuda"),
    )


def _recv_checkpoint_with_logging(
    *,
    transport: PGTransport,
    src_rank: int,
    step: int,
    timeout: timedelta,
) -> object:
    """Re-implementation of torchft.pg_transport.PGTransport.recv_checkpoint with per-tensor RSS logging.

    Same logic as upstream (commit reference: torchft pg_transport.py recv_checkpoint),
    but logs RSS / CUDA mem after every Nth tensor recv, plus final state at the end.
    """
    pg = transport._pg
    device = transport._device
    state_dict = transport._state_dict() if transport._state_dict else {}
    state_dict_leaves, _ = tree_flatten_with_path(state_dict)
    dst_tensors: dict = dict(state_dict_leaves)

    _log_mem("recv_loop.before_metadata")
    len_t = torch.zeros(1, dtype=torch.int64, device=device)
    pg.recv([len_t], src_rank, tag=1).wait(timeout)
    length = cast(int, len_t.item())
    assert length > 0, f"invalid metadata length {length=}"
    buf = torch.empty(length, dtype=torch.uint8, device=device)
    pg.recv([buf], src_rank, tag=2).wait(timeout)
    meta: _StateDictMeta = pickle.loads(buf.cpu().numpy().tobytes())
    assert meta.step == step

    _log_mem("recv_loop.after_metadata")
    n_tensors = sum(1 for v in meta.non_tensor_leaves if isinstance(v, (_TensorMeta, _DTensorMeta)))
    logger.info("[OOM_DEBUG] recv_loop.metadata: n_tensor_leaves=%d, n_total_leaves=%d", n_tensors, len(meta.non_tensor_leaves))

    i = 0
    works = []
    log_interval = max(n_tensors // 20, 1)  # ~20 progress lines

    def recv(path, v):
        nonlocal i
        inplace = dst_tensors.get(path)
        if isinstance(inplace, torch.Tensor) and inplace.device.type == device.type:
            if isinstance(inplace, DTensor):
                inplace = inplace._local_tensor
            t = _cast_tensor(inplace, torch.uint8)
            assert t.nbytes == v.nbytes
        else:
            t = torch.empty(v.nbytes, dtype=torch.uint8, device=device)
        work = pg.recv([t], src_rank, tag=3 + i)
        cur_i = i
        i += 1
        if inplace is None:
            work.wait(timeout)
            t = t.cpu()
            if cur_i % log_interval == 0 or cur_i == n_tensors - 1:
                _log_mem(f"recv_loop.tensor_{cur_i}_done_nbytes_{v.nbytes}")
                # Hypothesis: PyTorch pinned-host cache + glibc fragmentation accumulate
                # during the t.cpu() loop. Drain caches and gc to expose true RSS.
                torch._C._host_emptyCache()
                gc.collect()
                _log_mem(f"recv_loop.tensor_{cur_i}_after_emptyCache")
        else:
            works.append(work)
        return torch.as_strided(
            t.view(v.dtype),
            size=v.shape,
            stride=v.stride,
            storage_offset=v.storage_offset,
        )

    values = []
    for path, v in zip(meta.paths, meta.non_tensor_leaves):
        if isinstance(v, _TensorMeta):
            values.append(recv(path, v))
        elif isinstance(v, _DTensorMeta):
            tensor = recv(path, v.local)
            values.append(DTensor(tensor, v.spec, requires_grad=False))
        else:
            values.append(v)

    _log_mem("recv_loop.after_all_recv")
    for work in works:
        work.wait(timeout)
    _log_mem("recv_loop.after_works_wait")

    result = tree_unflatten(values, meta.treespec)
    _log_mem("recv_loop.after_tree_unflatten")
    gc.collect()
    torch._C._host_emptyCache()
    _log_mem("recv_loop.after_gc_and_host_emptyCache")
    return result
