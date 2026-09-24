from collections.abc import Callable
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from packaging.version import parse
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)

from miles.utils.ft_utils.process_group_utils import GeneralPGUtil

GLOO_GROUP = None

_T = TypeVar("_T")


def init_gloo_group():
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo")
    return GLOO_GROUP


def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    return GLOO_GROUP


class RankFailureError(RuntimeError):
    """Raised on every rank when a rank-local step failed on at least one rank."""


# torch.distributed.checkpoint raises CheckpointException, which derives from BaseException, not Exception.
RANK_LOCAL_ERRORS = (Exception, CheckpointException)


def raise_if_any_rank_failed(step: str, local_error: BaseException | None) -> None:
    """Collectively share rank-local errors over gloo so no rank waits on a collective a failed peer skips.

    Every rank must call. Without a process group the local error is re-raised unchanged.
    """
    if not dist.is_initialized():
        if local_error is not None:
            raise local_error
        return
    errors: list[str | None] = [None] * dist.get_world_size()
    local_message = None if local_error is None else f"{type(local_error).__name__}: {str(local_error).strip()}"
    dist.all_gather_object(errors, local_message, group=get_gloo_group())
    if any(errors):
        details = "; ".join(f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None)
        raise RankFailureError(f"{step} failed ({details})") from local_error


def run_on_all_ranks(step: str, operation: Callable[..., _T], *args, **kwargs) -> _T:
    """Every rank runs the operation; every rank raises if it failed on any rank."""
    result, local_error = None, None
    try:
        result = operation(*args, **kwargs)
    except RANK_LOCAL_ERRORS as error:
        local_error = error
    raise_if_any_rank_failed(step, local_error)
    return result


def run_on_rank0(step: str, operation: Callable[..., None], *args, **kwargs) -> None:
    """Rank 0 runs the operation; every rank raises if it failed there."""
    local_error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            operation(*args, **kwargs)
        except RANK_LOCAL_ERRORS as error:
            local_error = error
    raise_if_any_rank_failed(step, local_error)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if parse(torch.__version__) >= parse("2.6") else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def distributed_masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    shift_mean: bool = True,
    epsilon: float = 1e-8,
):
    """
    Performs whitening on a tensor using global statistics from all participating GPUs.

    It calculates the global mean and variance across all ranks in the default
    process group (the WORLD) and uses these global statistics to normalize the
    local data on each rank.

    Args:
        values (torch.Tensor): The local tensor of values to whiten.
        mask (torch.Tensor): The local mask corresponding to the values.
        process_group: The process group for all_reduce.
                      If None, uses the default world group.
        shift_mean (bool): If True, the output is zero-mean. Defaults to True.
        epsilon (float): A small value for numerical stability.

    Returns:
        torch.Tensor: The locally whitened tensor using global statistics.
    """
    # Calculate local intermediate statistics
    local_sum = (values * mask).sum()
    local_sum_sq = ((values**2) * mask).sum()
    local_mask_sum = mask.sum()

    stats_tensor = torch.tensor(
        [local_sum, local_sum_sq, local_mask_sum],
        device=values.device,
        dtype=torch.float32,
    )

    # Aggregate via all_reduce within the DP group
    GeneralPGUtil.create(process_group).all_reduce(stats_tensor, process_group, op=dist.ReduceOp.SUM)

    # Calculate global stats from aggregated results
    global_sum, global_sum_sq, global_mask_sum = stats_tensor

    if global_mask_sum.item() == 0:
        raise ValueError("The global mask sum across all participating GPUs is zero.")

    global_mean = global_sum / global_mask_sum
    global_mean_sq = global_sum_sq / global_mask_sum
    global_var = global_mean_sq - global_mean**2

    # Bessel's correction for unbiased estimate
    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction

    # Whiten local data using global stats
    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)

    if not shift_mean:
        whitened_values += global_mean

    return whitened_values
