"""Naming and topology helpers shared by the single- and multi-LoRA checkpoint paths."""

from pathlib import Path

import torch.distributed as dist

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group

# Cached by adapter_shard_topology(); the topology is fixed for the run.
_shard_topology: tuple[bool, tuple[tuple[int, int, int, int], ...]] | None = None


def raise_if_any_rank_failed(local_error: Exception | None, operation: str) -> None:
    """Raise on every rank when any rank reports ``local_error``, so failures stay collective."""
    message = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    if dist.is_initialized():
        group = get_gloo_group()
        messages: list[str | None] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(messages, message, group=group)
        message = next((item for item in messages if item is not None), None)

    if message is not None:
        error = RuntimeError(f"{operation} failed on at least one rank: {message}")
        if local_error is not None:
            raise error from local_error
        raise error


def megatron_shard_name(
    tp_rank: int,
    pp_rank: int,
    ep_rank: int = 0,
    etp_rank: int = 0,
    *,
    ep_size: int = 1,
    etp_size: int = 1,
    tp_size: int = 1,
) -> str:
    """Adapter shard name for one ``(tp, pp, ep, etp)`` coordinate; EP ranks hold different
    local experts and ETP ranks hold different slices of them. The ep suffix is omitted at
    ``ep_size == 1``, and the etp suffix at ``etp_size <= tp_size`` where the etp rank is a
    function of the tp rank, so checkpoints written before either existed stay loadable."""
    name = f"adapter_megatron_tp{tp_rank}_pp{pp_rank}"
    if ep_size > 1:
        name += f"_ep{ep_rank}"
    if etp_size > tp_size:
        name += f"_etp{etp_rank}"
    return name + ".pt"


def local_shard_name() -> str:
    """Shard name for this rank's model-parallel coordinate."""
    parallel_state = get_parallel_state()
    return megatron_shard_name(
        parallel_state.tp.rank,
        parallel_state.pp.rank,
        parallel_state.ep.rank,
        parallel_state.etp.rank,
        ep_size=parallel_state.ep.size,
        etp_size=parallel_state.etp.size,
        tp_size=parallel_state.tp.size,
    )


def adapter_shard_topology() -> tuple[bool, tuple[tuple[int, int, int, int], ...]]:
    """Return ``(this_rank_writes_its_shard, realized (tp, pp, ep, etp) coords)`` via one cached gloo all-gather."""
    global _shard_topology
    if _shard_topology is not None:
        return _shard_topology

    parallel_state = get_parallel_state()
    coords = (parallel_state.tp.rank, parallel_state.pp.rank, parallel_state.ep.rank, parallel_state.etp.rank)
    if not dist.is_initialized():
        _shard_topology = (True, (coords,))
        return _shard_topology

    current_rank = dist.get_rank()
    group = get_gloo_group()
    gathered: list[object] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, (coords, current_rank), group=group)
    is_writer = current_rank == min(rank for entry_coords, rank in gathered if entry_coords == coords)
    _shard_topology = (is_writer, tuple(sorted({entry_coords for entry_coords, _ in gathered})))
    return _shard_topology


def all_megatron_checkpoints_exist(step_dir: Path, shard_names) -> bool:
    return all((step_dir / name).exists() for name in shard_names)
